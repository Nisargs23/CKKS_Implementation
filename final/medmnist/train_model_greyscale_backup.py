"""
Train a CryptoNets-style Neural Network on MedMNIST for Encrypted Inference
============================================================================

This script:
1. Downloads BreastMNIST (64x64 breast ultrasound, binary: benign/malignant)
2. Trains a 5-layer network using x² activations (FHE-compatible)
3. Exports weights to a simple text format for C++ consumption
4. Also exports test images for encrypted inference testing

Architecture matches the C++ encrypted inference pipeline:
  Input (4096) → FC1 → x² → FC2 → x² → FC3 → x² → [BOOTSTRAP] → FC4 → x² → FC5 → Output

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
DATASET = "breastmnist"       # Binary classification, 64x64 ultrasound
IMAGE_SIZE = 64 * 64          # 4096 pixels (flattened)
NUM_CLASSES = 2               # benign / malignant
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
DEVICE = "cpu"                # CPU is fine for this small model

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(OUTPUT_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

# ============================================================
# Square Activation Function (CryptoNets-compatible)
# ============================================================
class SquareActivation(nn.Module):
    """f(x) = x² — the standard FHE-friendly activation function.
    
    This replaces ReLU/sigmoid because:
    - ReLU requires comparison (not possible in FHE)
    - Sigmoid requires high-degree polynomial approximation
    - x² is a single homomorphic multiplication
    """
    def forward(self, x):
        return x * x


# ============================================================
# CryptoNets-Style Network
# ============================================================
class CryptoNet(nn.Module):
    """5-layer element-wise neural network for encrypted inference.
    
    Each layer is a diagonal (element-wise) linear transformation:
      output[i] = weight[i] * input[i] + bias[i]
    
    This maps directly to CKKS SIMD operations:
      - EvalMult(ct, weight_plaintext)  →  element-wise multiply
      - EvalAdd(ct, bias_plaintext)     →  element-wise add
    
    Architecture:
      Layer 1: 4096 → 4096 (element-wise) + x²
      Layer 2: 4096 → 4096 (element-wise) + x²  
      Layer 3: 4096 → 4096 (element-wise) + x²
      [BOOTSTRAP would happen here in encrypted version]
      Layer 4: 4096 → 4096 (element-wise) + x²
      Layer 5: 4096 → 4096 (element-wise, output)
      
      Then: sum all 4096 outputs → single score → sigmoid → class
    """
    
    def __init__(self, input_size=IMAGE_SIZE):
        super(CryptoNet, self).__init__()
        
        self.input_size = input_size
        
        # Element-wise (diagonal) weight layers — each has input_size weights + input_size biases
        # We use nn.Linear with a trick: diagonal only
        # Actually, for true element-wise, we use learnable weight & bias vectors
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
        # But we include it here for training
        self.classifier = nn.Linear(input_size, NUM_CLASSES)
        
        self.act = SquareActivation()
        
        # Initialize weights carefully for x² stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with small values to prevent explosion through x² layers."""
        # x² makes values grow: if input ~0.5, after x² it's 0.25
        # After 4 layers of x², values shrink toward 0 for small inputs
        # and explode for inputs > 1. We need weights < 1 to keep things stable.
        for param in [self.w1, self.w2, self.w3, self.w4, self.w5]:
            nn.init.uniform_(param, 0.7, 1.1)
        for param in [self.b1, self.b2, self.b3, self.b4, self.b5]:
            nn.init.uniform_(param, -0.02, 0.02)
    
    def forward_elementwise(self, x):
        """Forward pass through element-wise layers only.
        Returns the 4096-dim feature vector before classification.
        This is what the encrypted C++ code computes.
        """
        # Layer 1 + x²
        x = self.w1 * x + self.b1
        x = self.act(x)
        
        # Layer 2 + x²
        x = self.w2 * x + self.b2
        x = self.act(x)
        
        # Layer 3 + x²
        x = self.w3 * x + self.b3
        x = self.act(x)
        
        # [BOOTSTRAP would happen here in encrypted version]
        
        # Layer 4 + x²
        x = self.w4 * x + self.b4
        x = self.act(x)
        
        # Layer 5 (no activation - output layer)
        x = self.w5 * x + self.b5
        
        return x
    
    def forward(self, x):
        """Full forward pass including classification head."""
        # Flatten image
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
    print("=" * 60)
    print("  Training CryptoNets Model on BreastMNIST")
    print("=" * 60)
    
    # Dataset info
    info = INFO[DATASET]
    n_channels = info["n_channels"]  # 1 for grayscale
    n_classes = len(info["label"])
    print(f"\nDataset: {DATASET}")
    print(f"  Task: {info['task']}")
    print(f"  Classes: {n_classes} ({', '.join(info['label'].values())})")
    print(f"  Channels: {n_channels}")
    
    # Download dataset
    DataClass = getattr(medmnist, info["python_class"])
    
    # Simple transform: convert to tensor, normalize to [0, 1]
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 255] → [0, 1]
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
    model = CryptoNet(IMAGE_SIZE).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    print(f"  Element-wise weights: {5 * IMAGE_SIZE:,} (5 layers × {IMAGE_SIZE})")
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
            
            # Gradient clipping to prevent x² explosion
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
    
    Format: weights_layer{N}.txt — one weight per line (4096 values)
             biases_layer{N}.txt — one bias per line (4096 values)
             classifier_weights.txt — 4096 x NUM_CLASSES (row-major)
             classifier_bias.txt — NUM_CLASSES values
    """
    print("\nExporting weights for C++ inference...")
    
    weights_dir = os.path.join(output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    # Export element-wise weights and biases
    for i, (wname, bname) in enumerate([
        ("w1", "b1"), ("w2", "b2"), ("w3", "b3"), ("w4", "b4"), ("w5", "b5")
    ], 1):
        w = getattr(model, wname).detach().cpu().numpy()
        b = getattr(model, bname).detach().cpu().numpy()
        
        np.savetxt(os.path.join(weights_dir, f"weights_layer{i}.txt"), w, fmt="%.10f")
        np.savetxt(os.path.join(weights_dir, f"biases_layer{i}.txt"), b, fmt="%.10f")
        
        print(f"  Layer {i}: weights range [{w.min():.4f}, {w.max():.4f}], "
              f"bias range [{b.min():.4f}, {b.max():.4f}]")
    
    # Export classifier (post-decryption aggregation)
    clf_w = model.classifier.weight.detach().cpu().numpy()  # [NUM_CLASSES, IMAGE_SIZE]
    clf_b = model.classifier.bias.detach().cpu().numpy()    # [NUM_CLASSES]
    
    np.savetxt(os.path.join(weights_dir, "classifier_weights.txt"), clf_w, fmt="%.10f")
    np.savetxt(os.path.join(weights_dir, "classifier_bias.txt"), clf_b, fmt="%.10f")
    
    print(f"  Classifier: {clf_w.shape[0]} classes × {clf_w.shape[1]} features")
    
    # Export metadata
    with open(os.path.join(weights_dir, "model_info.txt"), "w") as f:
        f.write(f"dataset={DATASET}\n")
        f.write(f"image_size={IMAGE_SIZE}\n")
        f.write(f"image_width=64\n")
        f.write(f"image_height=64\n")
        f.write(f"num_layers=5\n")
        f.write(f"num_classes={NUM_CLASSES}\n")
        f.write(f"activation=square\n")
        f.write(f"layers_before_bootstrap=3\n")
        f.write(f"layers_after_bootstrap=2\n")
    
    print(f"  Metadata saved to {weights_dir}/model_info.txt")
    print(f"  Total files: {2*5 + 3} files in {weights_dir}/")


# ============================================================
# Export Test Images for C++ 
# ============================================================
def export_test_images(test_dataset, output_dir, num_images=10):
    """Export test images as text files for C++ loading.
    
    Also exports as PGM for visual inspection.
    """
    print(f"\nExporting {num_images} test images for C++ inference...")
    
    images_dir = os.path.join(output_dir, "test_images")
    os.makedirs(images_dir, exist_ok=True)
    
    class_names = INFO[DATASET]["label"]
    
    for i in range(min(num_images, len(test_dataset))):
        image, label = test_dataset[i]
        label = int(label.squeeze())
        
        # image is [1, 64, 64] tensor in [0, 1]
        pixels = image.squeeze().numpy().flatten()  # [4096]
        
        # Save as text (one pixel per line)
        np.savetxt(os.path.join(images_dir, f"test_image_{i:03d}.txt"), pixels, fmt="%.10f")
        
        # Save as PGM for viewing
        pgm_pixels = (pixels * 255).astype(np.uint8)
        with open(os.path.join(images_dir, f"test_image_{i:03d}.pgm"), "w") as f:
            f.write(f"P2\n64 64\n255\n")
            for j, p in enumerate(pgm_pixels):
                f.write(f"{p}")
                if (j + 1) % 16 == 0:
                    f.write("\n")
                else:
                    f.write(" ")
        
        # Save label
        with open(os.path.join(images_dir, f"test_image_{i:03d}_label.txt"), "w") as f:
            f.write(f"{label}\n")
            f.write(f"{class_names[str(label)]}\n")
        
        print(f"  Image {i:3d}: label={label} ({class_names[str(label)]}), "
              f"mean={pixels.mean():.4f}, range=[{pixels.min():.4f}, {pixels.max():.4f}]")
    
    # Save class names
    with open(os.path.join(images_dir, "class_names.txt"), "w") as f:
        for k, v in class_names.items():
            f.write(f"{k} {v}\n")
    
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
        
        x = image.view(1, -1)  # [1, 4096]
        
        print(f"\n--- Test Image {i} (true label: {label} = {class_names[str(label)]}) ---")
        print(f"  Input:  mean={x.mean():.6f}, std={x.std():.6f}")
        
        with torch.no_grad():
            # Layer 1
            x_out = model.w1 * x + model.b1
            x_out = x_out * x_out  # x²
            print(f"  After Layer 1 + x²: mean={x_out.mean():.6f}, std={x_out.std():.6f}, "
                  f"range=[{x_out.min():.6f}, {x_out.max():.6f}]")
            
            # Layer 2
            x_out = model.w2 * x_out + model.b2
            x_out = x_out * x_out
            print(f"  After Layer 2 + x²: mean={x_out.mean():.6f}, std={x_out.std():.6f}, "
                  f"range=[{x_out.min():.6f}, {x_out.max():.6f}]")
            
            # Layer 3
            x_out = model.w3 * x_out + model.b3
            x_out = x_out * x_out
            print(f"  After Layer 3 + x²: mean={x_out.mean():.6f}, std={x_out.std():.6f}, "
                  f"range=[{x_out.min():.6f}, {x_out.max():.6f}]")
            
            # [BOOTSTRAP]
            
            # Layer 4
            x_out = model.w4 * x_out + model.b4
            x_out = x_out * x_out
            print(f"  After Layer 4 + x²: mean={x_out.mean():.6f}, std={x_out.std():.6f}, "
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
            print(f"  {'✓ CORRECT' if pred == label else '✗ WRONG'}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("CryptoNets Training Pipeline for MedMNIST")
    print(f"Dataset: {DATASET} (64×64 breast ultrasound)")
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
    print(f"  Test Accuracy: {test_acc:.1f}%")
    print(f"  Weights exported to: {OUTPUT_DIR}/weights/")
    print(f"  Test images exported to: {OUTPUT_DIR}/test_images/")
    print(f"  Run C++ encrypted inference with: ckks_medmnist.exe")
    print("=" * 60)
