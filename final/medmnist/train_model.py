"""
Train a CryptoNets-style Neural Network on MedMNIST for Encrypted Inference
============================================================================

This script:
1. Downloads a MedMNIST dataset (greyscale or RGB, configurable)
2. Trains a 5-layer network using x² activations (FHE-compatible)
3. Exports weights to a simple text format for C++ consumption
4. Also exports test images for encrypted inference testing

Supported datasets:
  - breastmnist   (64x64, 1-channel greyscale, binary: benign/malignant)
  - pathmnist     (64x64, 3-channel RGB, 9-class colon pathology)
  - dermamnist    (64x64, 3-channel RGB, 7-class dermatoscopy)
  - bloodmnist    (64x64, 3-channel RGB, 8-class blood cell)
  - retinamnist   (64x64, 3-channel RGB, 5-class retinal OCT)
  - organamnist   (64x64, 1-channel greyscale, 11-class organ)

Architecture matches the C++ encrypted inference pipeline:
  Input (C*4096) -> FC1 -> x^2 -> FC2 -> x^2 -> FC3 -> x^2
  -> [BOOTSTRAP] -> FC4 -> x^2 -> FC5 -> Output

For RGB images, each channel is processed independently with per-channel
weights. In the encrypted C++ pipeline, each channel gets its own
ciphertext (since 3*4096 = 12288 > 8192 max CKKS slots).

Note: "FC" here means element-wise (diagonal) linear layers to match
      the CKKS SIMD packing structure. Each "weight" is per-pixel.

Reference: CryptoNets (Gilad-Bachrach et al., ICML 2016)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO

# ============================================================
# Configuration
# ============================================================
# Choose dataset: "breastmnist" (greyscale), "pathmnist" (RGB),
#                 "dermamnist" (RGB), "bloodmnist" (RGB), etc.
DATASET = os.environ.get("MEDMNIST_DATASET", "breastmnist")

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
DEVICE = "cpu"                # CPU is fine for this small model

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(OUTPUT_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

# These are set dynamically based on dataset
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
CHANNEL_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT  # 4096 pixels per channel
NUM_CHANNELS = None  # Set from dataset info
NUM_CLASSES = None    # Set from dataset info
IMAGE_SIZE = None     # CHANNEL_SIZE * NUM_CHANNELS


# ============================================================
# Square Activation Function (CryptoNets-compatible)
# ============================================================
class SquareActivation(nn.Module):
    """f(x) = x^2 -- the standard FHE-friendly activation function.

    This replaces ReLU/sigmoid because:
    - ReLU requires comparison (not possible in FHE)
    - Sigmoid requires high-degree polynomial approximation
    - x^2 is a single homomorphic multiplication
    """
    def forward(self, x):
        return x * x


# ============================================================
# CryptoNets-Style Network (supports greyscale and RGB)
# ============================================================
class CryptoNet(nn.Module):
    """5-layer element-wise neural network for encrypted inference.

    Supports both greyscale (1 channel) and RGB (3 channels).

    For greyscale: input_size = 4096, one element-wise weight per pixel
    For RGB:       input_size = 3 * 4096 = 12288, per-channel weights
                   In FHE mode, each channel is encrypted separately

    Architecture:
      Layer 1: input_size -> input_size (element-wise) + x^2
      Layer 2: input_size -> input_size (element-wise) + x^2
      Layer 3: input_size -> input_size (element-wise) + x^2
      [BOOTSTRAP would happen here in encrypted version]
      Layer 4: input_size -> input_size (element-wise) + x^2
      Layer 5: input_size -> input_size (element-wise, output)

      Then: classifier (linear layer) -> class scores
    """

    def __init__(self, input_size, num_classes, num_channels=1):
        super(CryptoNet, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.channel_size = input_size // num_channels  # 4096 per channel

        # Element-wise (diagonal) weight layers
        # Each has input_size weights + input_size biases
        self.w1 = nn.Parameter(torch.ones(input_size) * 0.95)
        self.b1 = nn.Parameter(torch.zeros(input_size))

        self.w2 = nn.Parameter(torch.ones(input_size) * 0.92)
        self.b2 = nn.Parameter(torch.zeros(input_size))

        self.w3 = nn.Parameter(torch.ones(input_size) * 0.94)
        self.b3 = nn.Parameter(torch.zeros(input_size))

        self.w4 = nn.Parameter(torch.ones(input_size) * 0.90)
        self.b4 = nn.Parameter(torch.zeros(input_size))

        self.w5 = nn.Parameter(torch.ones(input_size) * 1.00)
        self.b5 = nn.Parameter(torch.zeros(input_size))

        # Aggregation: sum over all pixels then classify
        # This is done post-decryption in the encrypted pipeline
        self.classifier = nn.Linear(input_size, num_classes)

        self.act = SquareActivation()

        # Initialize weights carefully for x^2 stability
        self._init_weights()

    def _init_weights(self):
        """Initialize with small values to prevent explosion through x^2 layers."""
        for param in [self.w1, self.w2, self.w3, self.w4, self.w5]:
            nn.init.uniform_(param, 0.7, 1.1)
        for param in [self.b1, self.b2, self.b3, self.b4, self.b5]:
            nn.init.uniform_(param, -0.02, 0.02)

    def forward_elementwise(self, x):
        """Forward pass through element-wise layers only.
        Returns the feature vector before classification.
        This is what the encrypted C++ code computes.
        """
        # Layer 1 + x^2
        x = self.w1 * x + self.b1
        x = self.act(x)

        # Layer 2 + x^2
        x = self.w2 * x + self.b2
        x = self.act(x)

        # Layer 3 + x^2
        x = self.w3 * x + self.b3
        x = self.act(x)

        # [BOOTSTRAP would happen here in encrypted version]

        # Layer 4 + x^2
        x = self.w4 * x + self.b4
        x = self.act(x)

        # Layer 5 (no activation - output layer)
        x = self.w5 * x + self.b5

        return x

    def forward(self, x):
        """Full forward pass including classification head."""
        # Flatten image: [batch, C, 64, 64] -> [batch, C*64*64]
        x = x.view(-1, self.input_size)

        # Element-wise layers (this is the encrypted part)
        features = self.forward_elementwise(x)

        # Classification head (done post-decryption)
        logits = self.classifier(features)
        return logits


# ============================================================
# Training
# ============================================================
def train_model():
    global NUM_CHANNELS, NUM_CLASSES, IMAGE_SIZE

    print("=" * 60)
    print(f"  Training CryptoNets Model on {DATASET}")
    print("=" * 60)

    # Dataset info
    info = INFO[DATASET]
    NUM_CHANNELS = info["n_channels"]
    NUM_CLASSES = len(info["label"])
    IMAGE_SIZE = CHANNEL_SIZE * NUM_CHANNELS

    print(f"\nDataset: {DATASET}")
    print(f"  Task: {info['task']}")
    print(f"  Classes: {NUM_CLASSES} ({', '.join(info['label'].values())})")
    print(f"  Channels: {NUM_CHANNELS} ({'RGB' if NUM_CHANNELS == 3 else 'Greyscale'})")
    print(f"  Image size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}x{NUM_CHANNELS} = {IMAGE_SIZE} values")
    print(f"  Per-channel: {CHANNEL_SIZE} pixels")

    if NUM_CHANNELS == 3:
        print(f"\n  [RGB MODE] Each channel will be encrypted separately in FHE.")
        print(f"  Weights are organized as: [R_weights | G_weights | B_weights]")
        print(f"  Total weights per layer: {IMAGE_SIZE}")

    # Download dataset
    DataClass = getattr(medmnist, info["python_class"])

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 255] -> [0, 1], shape [C, H, W]
    ])

    train_dataset = DataClass(split="train", transform=transform, download=True,
                               root=DATASET_DIR, size=64)
    val_dataset = DataClass(split="val", transform=transform, download=True,
                             root=DATASET_DIR, size=64)
    test_dataset = DataClass(split="test", transform=transform, download=True,
                              root=DATASET_DIR, size=64)

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    model = CryptoNet(IMAGE_SIZE, NUM_CLASSES, NUM_CHANNELS).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    print(f"  Element-wise weights: {5 * IMAGE_SIZE:,} (5 layers x {IMAGE_SIZE})")
    print(f"  Element-wise biases:  {5 * IMAGE_SIZE:,}")
    print(f"  Classifier weights:   {IMAGE_SIZE * NUM_CLASSES + NUM_CLASSES:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training loop
    best_val_acc = 0.0
    print(f"\nTraining for {EPOCHS} epochs...")
    print("-" * 50)

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.squeeze().long().to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping to prevent x^2 explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        scheduler.step()

        train_loss /= train_total
        train_acc = 100. * train_correct / train_total

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.squeeze().long().to(DEVICE)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{EPOCHS}]  Loss: {train_loss:.4f}  "
                  f"Train Acc: {train_acc:.1f}%  Val Acc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "cryptonet_best.pth"))

    print("-" * 50)
    print(f"Best validation accuracy: {best_val_acc:.1f}%")

    # Load best model and test
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "cryptonet_best.pth"),
                                      weights_only=True))
    model.eval()

    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.squeeze().long().to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = 100. * test_correct / test_total
    print(f"Test accuracy:            {test_acc:.1f}%")

    return model, test_dataset, test_acc


# ============================================================
# Export Weights for C++
# ============================================================
def export_weights(model, output_dir):
    """Export trained weights in a simple text format for C++ loading.

    For greyscale (1 channel):
      weights_layer{N}.txt -- 4096 values (one weight per pixel per line)
      biases_layer{N}.txt  -- 4096 values

    For RGB (3 channels):
      weights_layer{N}.txt -- 12288 values (R_0..R_4095, G_0..G_4095, B_0..B_4095)
      biases_layer{N}.txt  -- 12288 values

      Additionally, per-channel weight files for C++ per-channel encryption:
      weights_layer{N}_ch{C}.txt -- 4096 values for channel C
      biases_layer{N}_ch{C}.txt  -- 4096 values for channel C
    """
    print(f"\nExporting weights for C++ inference ({NUM_CHANNELS} channel(s))...")

    weights_dir = os.path.join(output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Export element-wise weights and biases
    for i, (wname, bname) in enumerate([
        ("w1", "b1"), ("w2", "b2"), ("w3", "b3"), ("w4", "b4"), ("w5", "b5")
    ], 1):
        w = getattr(model, wname).detach().cpu().numpy()
        b = getattr(model, bname).detach().cpu().numpy()

        # Full weight vector (all channels concatenated)
        np.savetxt(os.path.join(weights_dir, f"weights_layer{i}.txt"), w, fmt="%.10f")
        np.savetxt(os.path.join(weights_dir, f"biases_layer{i}.txt"), b, fmt="%.10f")

        print(f"  Layer {i}: weights range [{w.min():.4f}, {w.max():.4f}], "
              f"bias range [{b.min():.4f}, {b.max():.4f}]")

        # For RGB: also export per-channel weight files
        if NUM_CHANNELS > 1:
            for ch in range(NUM_CHANNELS):
                ch_start = ch * CHANNEL_SIZE
                ch_end = ch_start + CHANNEL_SIZE
                w_ch = w[ch_start:ch_end]
                b_ch = b[ch_start:ch_end]
                np.savetxt(os.path.join(weights_dir, f"weights_layer{i}_ch{ch}.txt"),
                          w_ch, fmt="%.10f")
                np.savetxt(os.path.join(weights_dir, f"biases_layer{i}_ch{ch}.txt"),
                          b_ch, fmt="%.10f")
            print(f"    Per-channel files exported (ch0=R, ch1=G, ch2=B)")

    # Export classifier
    clf_w = model.classifier.weight.detach().cpu().numpy()  # [NUM_CLASSES, IMAGE_SIZE]
    clf_b = model.classifier.bias.detach().cpu().numpy()    # [NUM_CLASSES]

    np.savetxt(os.path.join(weights_dir, "classifier_weights.txt"), clf_w, fmt="%.10f")
    np.savetxt(os.path.join(weights_dir, "classifier_bias.txt"), clf_b, fmt="%.10f")

    print(f"  Classifier: {clf_w.shape[0]} classes x {clf_w.shape[1]} features")

    # For RGB: also export per-channel classifier weights
    if NUM_CHANNELS > 1:
        for ch in range(NUM_CHANNELS):
            ch_start = ch * CHANNEL_SIZE
            ch_end = ch_start + CHANNEL_SIZE
            clf_w_ch = clf_w[:, ch_start:ch_end]
            np.savetxt(os.path.join(weights_dir, f"classifier_weights_ch{ch}.txt"),
                      clf_w_ch, fmt="%.10f")
        print(f"  Per-channel classifier weights exported")

    # Export metadata
    with open(os.path.join(weights_dir, "model_info.txt"), "w") as f:
        f.write(f"dataset={DATASET}\n")
        f.write(f"image_size={IMAGE_SIZE}\n")
        f.write(f"channel_size={CHANNEL_SIZE}\n")
        f.write(f"image_width={IMAGE_WIDTH}\n")
        f.write(f"image_height={IMAGE_HEIGHT}\n")
        f.write(f"num_channels={NUM_CHANNELS}\n")
        f.write(f"num_layers=5\n")
        f.write(f"num_classes={NUM_CLASSES}\n")
        f.write(f"activation=square\n")
        f.write(f"layers_before_bootstrap=3\n")
        f.write(f"layers_after_bootstrap=2\n")
        f.write(f"color_mode={'rgb' if NUM_CHANNELS == 3 else 'greyscale'}\n")

    print(f"  Metadata saved to {weights_dir}/model_info.txt")

    total_files = 2 * 5 + 3  # base files
    if NUM_CHANNELS > 1:
        total_files += 2 * 5 * NUM_CHANNELS + NUM_CHANNELS  # per-channel files
    print(f"  Total files: {total_files} files in {weights_dir}/")


# ============================================================
# Export Test Images for C++
# ============================================================
def export_test_images(test_dataset, output_dir, num_images=10):
    """Export test images as text files for C++ loading.

    For greyscale: test_image_XXX.txt with 4096 values, test_image_XXX.pgm
    For RGB:       test_image_XXX.txt with 12288 values (R|G|B flattened)
                   test_image_XXX.ppm (RGB PPM file)
                   test_image_XXX_ch{0,1,2}.txt (per-channel, 4096 each)
    """
    print(f"\nExporting {num_images} test images for C++ inference "
          f"({NUM_CHANNELS} channel(s))...")

    images_dir = os.path.join(output_dir, "test_images")
    os.makedirs(images_dir, exist_ok=True)

    class_names = INFO[DATASET]["label"]

    for i in range(min(num_images, len(test_dataset))):
        image, label = test_dataset[i]
        label = int(label.squeeze())

        # image is [C, 64, 64] tensor in [0, 1]
        pixels_all = image.numpy()  # [C, 64, 64]

        # Flatten: [C, 64, 64] -> [C*64*64] (channel-first: R|G|B)
        pixels_flat = pixels_all.flatten()  # [C*4096]

        # Save as text (all channels concatenated, one value per line)
        np.savetxt(os.path.join(images_dir, f"test_image_{i:03d}.txt"),
                  pixels_flat, fmt="%.10f")

        if NUM_CHANNELS == 1:
            # Save as PGM (greyscale)
            pixels_1ch = pixels_all.squeeze()  # [64, 64]
            pgm_pixels = (pixels_1ch.flatten() * 255).astype(np.uint8)
            with open(os.path.join(images_dir, f"test_image_{i:03d}.pgm"), "w") as f:
                f.write(f"P2\n{IMAGE_WIDTH} {IMAGE_HEIGHT}\n255\n")
                for j, p in enumerate(pgm_pixels):
                    f.write(f"{p}")
                    if (j + 1) % 16 == 0:
                        f.write("\n")
                    else:
                        f.write(" ")
        else:
            # Save as PPM (RGB)
            # PPM P3 format: R G B R G B ...
            with open(os.path.join(images_dir, f"test_image_{i:03d}.ppm"), "w") as f:
                f.write(f"P3\n{IMAGE_WIDTH} {IMAGE_HEIGHT}\n255\n")
                for y in range(IMAGE_HEIGHT):
                    for x in range(IMAGE_WIDTH):
                        r = int(pixels_all[0, y, x] * 255)
                        g = int(pixels_all[1, y, x] * 255)
                        b = int(pixels_all[2, y, x] * 255)
                        f.write(f"{r} {g} {b} ")
                    f.write("\n")

            # Also save per-channel text files for C++ per-channel encryption
            for ch in range(NUM_CHANNELS):
                ch_pixels = pixels_all[ch].flatten()  # [4096]
                np.savetxt(
                    os.path.join(images_dir, f"test_image_{i:03d}_ch{ch}.txt"),
                    ch_pixels, fmt="%.10f"
                )

        # Save label
        with open(os.path.join(images_dir, f"test_image_{i:03d}_label.txt"), "w") as f:
            f.write(f"{label}\n")
            f.write(f"{class_names[str(label)]}\n")

        mean_val = pixels_flat.mean()
        if NUM_CHANNELS == 3:
            ch_means = [pixels_all[c].mean() for c in range(3)]
            print(f"  Image {i:3d}: label={label} ({class_names[str(label)]}), "
                  f"mean=R:{ch_means[0]:.3f}/G:{ch_means[1]:.3f}/B:{ch_means[2]:.3f}")
        else:
            print(f"  Image {i:3d}: label={label} ({class_names[str(label)]}), "
                  f"mean={mean_val:.4f}, range=[{pixels_flat.min():.4f}, {pixels_flat.max():.4f}]")

    # Save class names and metadata
    with open(os.path.join(images_dir, "class_names.txt"), "w") as f:
        for k, v in class_names.items():
            f.write(f"{k} {v}\n")

    with open(os.path.join(images_dir, "image_info.txt"), "w") as f:
        f.write(f"num_channels={NUM_CHANNELS}\n")
        f.write(f"width={IMAGE_WIDTH}\n")
        f.write(f"height={IMAGE_HEIGHT}\n")
        f.write(f"channel_size={CHANNEL_SIZE}\n")
        f.write(f"total_size={IMAGE_SIZE}\n")
        f.write(f"color_mode={'rgb' if NUM_CHANNELS == 3 else 'greyscale'}\n")

    print(f"  Saved to {images_dir}/")


# ============================================================
# Verify: Run plaintext inference and check consistency
# ============================================================
def verify_inference(model, test_dataset, num_samples=5):
    """Run inference step by step and print intermediate values.
    This helps verify the C++ encrypted version matches.
    """
    print("\n" + "=" * 60)
    print("  Verification: Step-by-step Plaintext Inference")
    print("=" * 60)

    model.eval()
    class_names = INFO[DATASET]["label"]

    for i in range(min(num_samples, len(test_dataset))):
        image, label = test_dataset[i]
        label = int(label.squeeze())

        x = image.view(1, -1)  # [1, C*4096]

        print(f"\n--- Test Image {i} (true label: {label} = {class_names[str(label)]}) ---")
        print(f"  Input:  mean={x.mean():.6f}, std={x.std():.6f}, "
              f"channels={NUM_CHANNELS}, size={x.shape[1]}")

        with torch.no_grad():
            # Layer 1
            x_out = model.w1 * x + model.b1
            x_out = x_out * x_out  # x^2
            print(f"  After Layer 1 + x^2: mean={x_out.mean():.6f}, std={x_out.std():.6f}, "
                  f"range=[{x_out.min():.6f}, {x_out.max():.6f}]")

            # Layer 2
            x_out = model.w2 * x_out + model.b2
            x_out = x_out * x_out
            print(f"  After Layer 2 + x^2: mean={x_out.mean():.6f}, std={x_out.std():.6f}, "
                  f"range=[{x_out.min():.6f}, {x_out.max():.6f}]")

            # Layer 3
            x_out = model.w3 * x_out + model.b3
            x_out = x_out * x_out
            print(f"  After Layer 3 + x^2: mean={x_out.mean():.6f}, std={x_out.std():.6f}, "
                  f"range=[{x_out.min():.6f}, {x_out.max():.6f}]")

            # [BOOTSTRAP]

            # Layer 4
            x_out = model.w4 * x_out + model.b4
            x_out = x_out * x_out
            print(f"  After Layer 4 + x^2: mean={x_out.mean():.6f}, std={x_out.std():.6f}, "
                  f"range=[{x_out.min():.6f}, {x_out.max():.6f}]")

            # Layer 5 (no activation)
            x_out = model.w5 * x_out + model.b5
            print(f"  After Layer 5:      mean={x_out.mean():.6f}, std={x_out.std():.6f}, "
                  f"range=[{x_out.min():.6f}, {x_out.max():.6f}]")

            # Classify
            logits = model.classifier(x_out)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()

            print(f"  Logits: {logits.squeeze().numpy()}")
            print(f"  Probs:  {probs.squeeze().numpy()}")
            print(f"  Predicted: {pred} ({class_names[str(pred)]})")
            print(f"  {'CORRECT' if pred == label else 'WRONG'}")

            if NUM_CHANNELS == 3:
                # Show per-channel statistics
                for ch, name in enumerate(['R', 'G', 'B']):
                    ch_start = ch * CHANNEL_SIZE
                    ch_end = ch_start + CHANNEL_SIZE
                    ch_out = x_out[0, ch_start:ch_end]
                    print(f"    Channel {name}: mean={ch_out.mean():.6f}, "
                          f"range=[{ch_out.min():.6f}, {ch_out.max():.6f}]")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # Allow dataset selection via command line
    if len(sys.argv) > 1:
        DATASET = sys.argv[1]

    if DATASET not in INFO:
        print(f"ERROR: Unknown dataset '{DATASET}'")
        print(f"Available: {list(INFO.keys())}")
        sys.exit(1)

    print("CryptoNets Training Pipeline for MedMNIST")
    print(f"Dataset: {DATASET}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device:  {DEVICE}")
    print()

    # Train
    model, test_dataset, test_acc = train_model()

    # Export
    export_weights(model, OUTPUT_DIR)
    export_test_images(test_dataset, OUTPUT_DIR, num_images=10)

    # Verify
    verify_inference(model, test_dataset, num_samples=5)

    print("\n" + "=" * 60)
    print(f"  TRAINING COMPLETE")
    print(f"  Dataset: {DATASET} ({NUM_CHANNELS} channel(s), "
          f"{'RGB' if NUM_CHANNELS == 3 else 'Greyscale'})")
    print(f"  Test Accuracy: {test_acc:.1f}%")
    print(f"  Weights exported to: {OUTPUT_DIR}/weights/")
    print(f"  Test images exported to: {OUTPUT_DIR}/test_images/")
    print(f"  Run C++ encrypted inference with: ckks_medmnist.exe")
    if NUM_CHANNELS == 3:
        print(f"  [RGB] Per-channel weight/image files available for FHE")
    print("=" * 60)
