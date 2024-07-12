# Define color variables
$GREEN = "DarkGreen"
$RED = "DarkRed"
$YELLOW = "DarkYellow"
$BLUE = "DarkBlue"
$CYAN = "Cyan"
$NC = "Default"

# Function to print messages with colors
function Write-Color {
    param (
        [string]$Text,
        [string]$Color,
        [switch]$NoNewline
    )
    if ($NoNewline) {
        Write-Host $Text -ForegroundColor $Color -NoNewline
    } else {
        Write-Host $Text -ForegroundColor $Color
    }
}

# Determine the script's directory
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

# Function to check if a command exists
function Command-Exists {
    param (
        [string]$Command
    )
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

# Function to get CUDA version using nvcc
function Get-CudaVersion {
    if (Command-Exists "nvcc") {
        $nvccOutput = nvcc --version | Select-String "release" | ForEach-Object { $_.ToString() }
        if ($nvccOutput -match "release (\d+\.\d+)") {
            return $matches[1]
        }
    }
    return $null
}

# Function to install JAX based on CUDA version
function Install-JAX {
    param (
        [string]$cudaVersion
    )

    Write-Color "CUDA version: $cudaVersion" $YELLOW

    switch -Regex ($cudaVersion) {
        "11\.\d+" {
            Write-Color "Detected CUDA 11. Installing JAX for CUDA 11..." $YELLOW
            conda run -n mmp python -m pip install "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
        }
        "12\.\d+" {
            Write-Color "Detected CUDA 12. Installing JAX for CUDA 12..." $YELLOW
            conda run -n mmp python -m pip install "jax[cuda120]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
        }
        default {
            Write-Color "No supported CUDA version detected. Installing CPU-only JAX..." $YELLOW
            conda run -n mmp python -m pip install jax jaxlib
        }
    }
    conda run -n mmp python -m pip install chex optax dm-haiku jraph
}

# Check if Conda is installed
if (-not (Command-Exists "conda")) {
    Write-Color "Conda is not installed. Please install Conda and try again." $RED
    exit 1
}

# Check if 7z is installed
if (-not (Command-Exists "7z")) {
    Write-Color "7z is not installed. Please install 7z and try again." $RED
    exit 1
}

# Auto Setup Wizard for Materials Modeling Project

# Step 0: Install Dependencies
function Install-Dependencies {
    if (conda env list | Select-String 'mmp') {
        Write-Color "Conda environment 'mmp' already exists. Skipping installation." $GREEN
    } else {
        Write-Color "Creating a new conda environment and installing dependencies..." $YELLOW
        conda env create -f "$SCRIPT_DIR\environment.yaml"
        Write-Color "To activate the environment, use 'conda activate mmp'" $GREEN

        # Install JAX based on CUDA version
        $cudaVersion = Get-CudaVersion
        Install-JAX $cudaVersion
    }
}

# Step 1: Create a Folder for the Experiment
function Create-Experiment-Folder {
    param (
        [string]$ExperimentDir
    )
    New-Item -ItemType Directory -Force -Path "$ExperimentDir\input", "$ExperimentDir\output" | Out-Null
    Copy-Item "$SCRIPT_DIR\includes.py" "$ExperimentDir"
}

# Step 2: Select a Model
function Select-Model {
    param (
        [string]$ExperimentDir
    )
    Copy-Item "$SCRIPT_DIR\MODELS\NNN.py" "$ExperimentDir\"
}

# Step 3: Extract POSCAR.7z if POSCAR directory is missing
function Extract-POSCAR {
    param (
        [string]$ExperimentDir
    )
    if (-not (Test-Path "$SCRIPT_DIR\POSCAR")) {
        Write-Color "POSCAR directory not found. Extracting POSCAR.7z..." $YELLOW
        7z x "$SCRIPT_DIR\POSCAR.7z" -o"$SCRIPT_DIR"
        cmd /c mklink /J "$ExperimentDir\POSCAR" "$SCRIPT_DIR\POSCAR"
        cmd /c mklink /J "$ExperimentDir\CSV" "$SCRIPT_DIR\CSV"
        Write-Color "POSCAR directory extracted and linked to experiment directory." $GREEN
    } else {
        cmd /c mklink /J "$ExperimentDir\POSCAR" "$SCRIPT_DIR\POSCAR"
        cmd /c mklink /J "$ExperimentDir\CSV" "$SCRIPT_DIR\CSV"
        Write-Color "POSCAR directory linked to experiment directory." $GREEN
    }
}

# Step 4: Prepare Input Components
function Prepare-Input-Components {
    param (
        [string]$ExperimentDir
    )
    Copy-Item "$SCRIPT_DIR\COMPONENTS\poscar_atomic_type_periodic_table.py" "$ExperimentDir\input\"
    Copy-Item "$SCRIPT_DIR\COMPONENTS\poscar_atomic_type_one_hot.py" "$ExperimentDir\input\"
    Copy-Item "$SCRIPT_DIR\COMPONENTS\space_group_one_hot.py" "$ExperimentDir\input\"
}

# Step 5: Prepare Target Component
function Prepare-Target-Component {
    param (
        [string]$ExperimentDir,
        [string]$TargetFile
    )
    Copy-Item "$SCRIPT_DIR\COMPONENTS\$TargetFile" "$ExperimentDir\output\"
}

# Step 6: Run the Experiment
function Run-Experiment {
    param (
        [string]$ExperimentDir,
        [string]$ExperimentName
    )
    conda run -n mmp python "$ExperimentDir\NNN.py" | Tee-Object -FilePath "run_$ExperimentName.txt"
}

# Step 7: Collect results
# Function to extract the highest validation accuracy
function Extract-Highest-Validation-Accuracy {
    param (
        [string]$File
    )
    Get-Content $File | Select-String 'Validation accuracy' | ForEach-Object {
        if ($_ -match 'Validation accuracy: \[([0-9.]+)\]') {
            $accuracy = [double]$matches[1]
            if ($accuracy -gt $global:max_accuracy) {
                $global:max_accuracy = $accuracy
                $global:max_line = $_
            }
        }
    }
    if ($global:max_line) {
        Write-Output $global:max_line
    } else {
        Write-Output " - "
    }
}

# Function to process all result files
function Process-Results {
    foreach ($result_file in Get-ChildItem -Filter "run_*.txt") {
        Write-Color "$($result_file): " $BLUE -NoNewline
        Extract-Highest-Validation-Accuracy $result_file
    }
}

# Main Execution
Write-Color "Starting experiment setup and execution..." $CYAN
Install-Dependencies

foreach ($target_file in Get-ChildItem "$SCRIPT_DIR\COMPONENTS\*.py") {
    if ($target_file.Name -notmatch "poscar_*") {
        $experiment_dir = "$SCRIPT_DIR\experiment_$($target_file.BaseName)"
        Write-Color "Starting experiment: $($target_file.Name)..." $CYAN
        Create-Experiment-Folder $experiment_dir
        Extract-POSCAR $experiment_dir
        Select-Model $experiment_dir
        Prepare-Input-Components $experiment_dir
        Prepare-Target-Component $experiment_dir $target_file.Name
        Run-Experiment $experiment_dir $target_file.Name
    }
}

Write-Color "All experiments have been set up and are running concurrently." $GREEN
Process-Results
Write-Color "All experiments have completed." $GREEN
