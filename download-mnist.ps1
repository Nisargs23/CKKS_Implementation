# Download MNIST Dataset Script
# Run this to download and extract MNIST files

$mnistDir = "D:\Manipal\Research\Implementation\data\mnist"
New-Item -ItemType Directory -Force -Path $mnistDir | Out-Null

Write-Host "Downloading MNIST dataset from alternative source..." -ForegroundColor Green

# Using ossci-datasets mirror (PyTorch's mirror)
$baseUrl = "https://ossci-datasets.s3.amazonaws.com/mnist"

$files = @(
    @{
        url = "$baseUrl/train-images-idx3-ubyte.gz"
        name = "train-images-idx3-ubyte.gz"
    },
    @{
        url = "$baseUrl/train-labels-idx1-ubyte.gz"
        name = "train-labels-idx1-ubyte.gz"
    },
    @{
        url = "$baseUrl/t10k-images-idx3-ubyte.gz"
        name = "t10k-images-idx3-ubyte.gz"
    },
    @{
        url = "$baseUrl/t10k-labels-idx1-ubyte.gz"
        name = "t10k-labels-idx1-ubyte.gz"
    }
)

foreach ($file in $files) {
    $outPath = Join-Path $mnistDir $file.name
    $extractedName = $file.name -replace '\.gz$', ''
    $extractedPath = Join-Path $mnistDir $extractedName
    
    if (Test-Path $extractedPath) {
        Write-Host "  $extractedName already exists, skipping..." -ForegroundColor Yellow
        continue
    }
    
    Write-Host "  Downloading $($file.name)..." -ForegroundColor Cyan
    try {
        Invoke-WebRequest -Uri $file.url -OutFile $outPath -UseBasicParsing
        
        # Extract using .NET
        Write-Host "  Extracting $($file.name)..." -ForegroundColor Cyan
        $inStream = [System.IO.File]::OpenRead($outPath)
        $gzipStream = New-Object System.IO.Compression.GzipStream($inStream, [System.IO.Compression.CompressionMode]::Decompress)
        $outStream = [System.IO.File]::Create($extractedPath)
        $gzipStream.CopyTo($outStream)
        $outStream.Close()
        $gzipStream.Close()
        $inStream.Close()
        
        # Remove .gz file
        Remove-Item $outPath
        Write-Host "  Extracted to $extractedName" -ForegroundColor Green
    }
    catch {
        Write-Host "  Error downloading $($file.name): $_" -ForegroundColor Red
    }
}

Write-Host "`nMNIST dataset ready in: $mnistDir" -ForegroundColor Green
Write-Host "`nTo run the MNIST example:" -ForegroundColor Yellow
Write-Host '  cd D:\Manipal\Research\Implementation\build\bin' -ForegroundColor White
Write-Host '  $env:Path = "D:\Manipal\Research\openfhe-development\build\lib;C:\msys64\mingw64\bin;$env:Path"' -ForegroundColor White
Write-Host '  .\ckks_bootstrap_mnist.exe "..\..\data\mnist\train-images-idx3-ubyte" "..\..\data\mnist\train-labels-idx1-ubyte" 0' -ForegroundColor White
