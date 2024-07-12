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

# ASCII Art Banner
function Print-Banner {
    Write-Color @"
????????????????????  ???  ?????????????  ?  ???????  ????????? ??????????
?????? ? ?? ????????  ???  ???? ? ???? ?  ?  ????? ?  ??????? ? ??? ?   ? 
? ?? ? ? ???????? ???????  ? ???????????????????????  ?  ?????????????? ? 
          ????????????????? ?????????????????  ? ??????????????           
          ?  ? ?????? ?? ?? ??????? ? ?? ????  ????????????? ??           
          ??????????  ??????????? ? ? ???????  ???????? ???????           
"@ $BLUE
}

# Function to check if a command exists
function Command-Exists {
    param (
        [string]$Command
    )
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
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

# Function to detect NVIDIA driver version and suggest CUDA version
function Detect-CudaVersion {
    $driverVersion = (nvidia-smi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1)
    if (-not $driverVersion) {
        Write-Color "No NVIDIA driver detected. Defaulting to CPU version of JAX." $YELLOW
        $global:cudaVersion = ""
    } else {
        Write-Color "Detected NVIDIA driver version: $driverVersion" $GREEN
        $driverParts = $driverVersion -split "\."
        $driverMajor = [int]$driverParts[0]
        $driverMinor = [int]$driverParts[1]

        if ($driverMajor -ge 450) {
            Write-Color "CUDA 12 is supported." $GREEN
            $global:cudaVersion = "cuda12"
        } else {
            Write-Color "Unsupported NVIDIA driver version. Please upgrade your driver to at least version 450." $RED
            exit 1
        }
    }
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
        Write-Color "Installing JAX and JAXLIB..." $YELLOW
        conda activate mmp
        conda run -n mmp pip install --upgrade pip
        Detect-CudaVersion
        if (-not $global:cudaVersion) {
            conda run -n mmp pip install jax chex optax dm-haiku jraph
        } else {
            conda run -n mmp pip install "jax[$global:cudaVersion]" chex optax dm-haiku jraph
        }
    }
}

# Step 1: Create a Folder for the Experiment
function Create-Experiment-Folder {
    while ($true) {
        Write-Color "Enter the name of your experiment directory: " $CYAN -NoNewline
        $experimentDir = Read-Host
        if (-not $experimentDir) {
            Write-Color "Experiment directory name cannot be empty. Please enter a valid name." $RED
        } elseif (Test-Path $experimentDir) {
            Write-Color "Experiment directory already exists. Do you want to overwrite it? (y/n): " $RED -NoNewline
            $overwrite = Read-Host
            if ($overwrite -eq "y") {
                Remove-Item -Recurse -Force $experimentDir
                New-Item -ItemType Directory -Path "$experimentDir\input", "$experimentDir\output" | Out-Null
                Write-Color "Experiment directory '$experimentDir' created with 'inputs' and 'target' subfolders." $GREEN
                break
            } elseif ($overwrite -eq "n") {
                while ($true) {
                    Write-Color "Do you want to run the existing experiment now? (y/n): " $CYAN -NoNewline
                    $runExisting = Read-Host
                    if ($runExisting -eq "y") {
                        Write-Color "Running the existing experiment..." $YELLOW
                        $modelFile = Get-ChildItem "$experimentDir" -Filter "*.py" | Where-Object { $_.Name -ne "includes.py" } | Select-Object -First 1
                        if ($modelFile) {
                            conda run -n mmp python $modelFile.FullName
                        } else {
                            Write-Color "No model script found in the existing experiment directory." $RED
                        }
                        exit 0
                    } elseif ($runExisting -eq "n") {
                        Write-Color "Please enter a different directory name." $YELLOW
                        break
                    } else {
                        Write-Color "Invalid response. Please enter 'y' for yes or 'n' for no." $RED
                    }
                }
            } else {
                Write-Color "Invalid response. Please enter 'y' for yes or 'n' for no." $RED
            }
        } else {
            New-Item -ItemType Directory -Path "$experimentDir\input", "$experimentDir\output" | Out-Null
            Write-Color "Experiment directory '$experimentDir' created with 'inputs' and 'target' subfolders." $GREEN
            break
        }
    }
    Copy-Item "$SCRIPT_DIR\includes.py" "$experimentDir"
}

# Step 2: Select a Model
function Select-Model {
    Write-Color "Available models:" $BLUE
    Get-ChildItem "$SCRIPT_DIR\MODELS"
    while ($true) {
        Write-Color "Enter the model filename you want to copy to your experiment directory: " $CYAN -NoNewline
        $modelFile = Read-Host
        if (Test-Path "$SCRIPT_DIR\MODELS\$modelFile") {
            Copy-Item "$SCRIPT_DIR\MODELS\$modelFile" "$experimentDir\"
            Write-Color "Model '$modelFile' copied to '$experimentDir'." $GREEN
            break
        } else {
            Write-Color "Model file does not exist. Please enter a valid filename from the list above." $RED
        }
    }
}

# Step 3: Extract POSCAR.7z if POSCAR directory is missing
function Extract-POSCAR {
    if (-not (Test-Path "$SCRIPT_DIR\POSCAR")) {
        Write-Color "POSCAR directory not found. Extracting POSCAR.7z..." $YELLOW
        7z x "$SCRIPT_DIR\POSCAR.7z" -o"$SCRIPT_DIR"
        cmd /c mklink /J "$experimentDir\POSCAR" "$SCRIPT_DIR\POSCAR"
        cmd /c mklink /J "$experimentDir\CSV" "$SCRIPT_DIR\CSV"
        Write-Color "POSCAR directory extracted and linked to experiment directory." $GREEN
    } else {
        cmd /c mklink /J "$experimentDir\POSCAR" "$SCRIPT_DIR\POSCAR"
        cmd /c mklink /J "$experimentDir\CSV" "$SCRIPT_DIR\CSV"
        Write-Color "POSCAR directory linked to experiment directory." $GREEN
    }
}

# Step 4: Prepare Input Components
function Prepare-Input-Components {
    Write-Color "Available components:" $BLUE
    Get-ChildItem "$SCRIPT_DIR\COMPONENTS"
    Write-Color "Copying selected components to the 'inputs' folder..." $YELLOW
    $poscarAtomicIncluded = $false
    while ($true) {
        Write-Color "Enter a component filename to copy (or 'done' to finish): " $CYAN -NoNewline
        $componentFile = Read-Host
        if ($componentFile -eq "done") {
            if (-not $poscarAtomicIncluded) {
                Write-Color "At least one poscar_atomic_ type component is required." $RED
            } else {
                break
            }
        } elseif (Test-Path "$SCRIPT_DIR\COMPONENTS\$componentFile") {
            Copy-Item "$SCRIPT_DIR\COMPONENTS\$componentFile" "$experimentDir\input\"
            Write-Color "Component '$componentFile' copied to '$experimentDir\input\'." $GREEN
            if ($componentFile -like "poscar_atomic_*") {
                $poscarAtomicIncluded = $true
            }
        } else {
            Write-Color "Component file does not exist. Please enter a valid filename from the list above." $RED
        }
    }
}

# Step 5: Prepare Target Component
function Prepare-Target-Component {
    Write-Color "Available components:" $BLUE
    Get-ChildItem "$SCRIPT_DIR\COMPONENTS"
    while ($true) {
        Write-Color "Enter the target component filename to copy: " $CYAN -NoNewline
        $targetFile = Read-Host
        if ($targetFile -notlike "poscar_*") {
            if (Test-Path "$SCRIPT_DIR\COMPONENTS\$targetFile") {
                Copy-Item "$SCRIPT_DIR\COMPONENTS\$targetFile" "$experimentDir\output\"
                Write-Color "Target component '$targetFile' copied to '$experimentDir\output\'." $GREEN
                break
            } else {
                Write-Color "Target component file does not exist. Please enter a valid filename from the list above." $RED
            }
        } else {
            Write-Color "Invalid target component. POSCAR files are not valid in the target folder. Please enter a different filename." $RED
        }
    }
}

# Step 6: Run the Experiment
function Run-Experiment {
    while ($true) {
        Write-Color "Do you want to run the experiment now? (y/n): " $CYAN -NoNewline
        $runNow = Read-Host
        if ($runNow -eq "y") {
            Write-Color "Running the experiment..." $YELLOW
            conda run -n mmp python "$experimentDir\$modelFile"
            break
        } elseif ($runNow -eq "n") {
            Write-Color "Setup completed. You can run your experiment from the '$experimentDir' directory." $GREEN
            break
        } else {
            Write-Color "Invalid response. Please enter 'y' for yes or 'n' for no." $RED
        }
    }
}

# Main Execution
Print-Banner
Write-Color "Welcome to the Auto Setup Wizard for the Materials Modeling Project!" $CYAN

Write-Color @"
(0/6)
??????????????  ?    ??????????????????????????????????
??????? ? ????  ?     ???? ????? ??? ???? ????  ??? ???
??????? ? ? ???????  ???????  ?????????????????????????
"@ $BLUE
Install-Dependencies

Write-Color @"
(1/6)
????? ???????????????????????  ???????  ?????????
?? ????????? ????????? ??? ?   ?? ? ??   ???? ???
???? ???  ???????? ??????? ?   ?  ???????????????
"@ $BLUE
Create-Experiment-Folder
Extract-POSCAR

Write-Color @"
(2/6)
???????  ?????????  ?????????????  
????? ?  ?? ?   ?   ???? ? ???? ?  
??????????????? ?   ? ?????????????
"@ $BLUE
Select-Model

Write-Color @"
(3/6)
???????  ?????????  ?????????????    ???????? ???????
????? ?  ?? ?   ?   ???? ? ???? ?    ???????? ? ? ???
??????????????? ?   ? ?????????????  ?????  ??? ? ???
"@ $BLUE
Prepare-Input-Components

Write-Color @"
(4/6)
???????  ?????????  ?????????????    ??????????????????
????? ?  ?? ?   ?   ???? ? ???? ?     ? ??????? ???  ? 
??????????????? ?   ? ?????????????   ? ? ?????????? ? 
"@ $BLUE
Prepare-Target-Component

Write-Color @"
(5/6)
????????????
 ??? ?????? 
????????????
"@ $BLUE
Run-Experiment

Write-Color "All steps completed successfully." $GREEN

# End of script
