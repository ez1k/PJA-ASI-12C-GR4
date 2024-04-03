$envName = "PJA-ASI-12C-GR4"
$envFile = "environment.yml"

if (-not (Test-Path $envFile)) {
    Write-Host "Error: '$envFile' not found."
    exit 1
}

conda env create -f $envFile
conda activate $envName

Write-Host "'$envName' aktywowany"