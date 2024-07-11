#!/bin/bash

# Define color variables
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Determine the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to check if a command exists
function command_exists {
    command -v "$1" >/dev/null 2>&1
}

# Check if Conda is installed
if ! command_exists conda; then
    echo -e "${RED}Conda is not installed. Please install Conda and try again.${NC}"
    exit 1
fi

# Check if 7z is installed
if ! command_exists 7z; then
    echo -e "${RED}7z is not installed. Please install 7z and try again.${NC}"
    exit 1
fi

# Auto Setup Wizard for Materials Modeling Project

# Step 0: Install Dependencies
function install_dependencies {
    if conda env list | grep -q 'mmp'; then
        echo -e "${GREEN}Conda environment 'mmp' already exists. Skipping installation.${NC}"
    else
        echo -e "${YELLOW}Creating a new conda environment and installing dependencies...${NC}"
        conda env create -f "$SCRIPT_DIR/environment.yaml"
        echo -e "${GREEN}To activate the environment, use 'conda activate mmp'${NC}"
        # Install JAX and JAXLIB separately
        echo -e "${YELLOW}Installing JAX and JAXLIB...${NC}"
        conda activate mmp
        conda run -n mmp pip install --upgrade pip
        conda run -n mmp pip install jax chex optax dm-haiku jraph
    fi
}

# Step 1: Create a Folder for the Experiment
function create_experiment_folder {
    local experiment_dir="$1"
    mkdir -p "$experiment_dir/input" "$experiment_dir/output"
    cp "$SCRIPT_DIR/includes.py" "$experiment_dir"
}

# Step 2: Select a Model
function select_model {
    local experiment_dir="$1"
    cp "$SCRIPT_DIR/MODELS/NNN.py" "$experiment_dir/"
}

# Step 3: Extract POSCAR.7z if POSCAR directory is missing
function extract_poscar {
    local experiment_dir="$1"
    if [[ ! -d "$SCRIPT_DIR/POSCAR" ]]; then
        echo -e "${YELLOW}POSCAR directory not found. Extracting POSCAR.7z...${NC}"
        7z x "$SCRIPT_DIR/POSCAR.7z" -o"$SCRIPT_DIR"
        ln -s "$SCRIPT_DIR/POSCAR" "$experiment_dir/POSCAR"
        ln -s "$SCRIPT_DIR/CSV" "$experiment_dir/CSV"
        echo -e "${GREEN}POSCAR directory extracted and linked to experiment directory.${NC}"
    else
        ln -s "$SCRIPT_DIR/POSCAR" "$experiment_dir/POSCAR"
        ln -s "$SCRIPT_DIR/CSV" "$experiment_dir/CSV"
        echo -e "${GREEN}POSCAR directory linked to experiment directory.${NC}"
    fi
}

# Step 4: Prepare Input Components
function prepare_input_components {
    local experiment_dir="$1"
    cp "$SCRIPT_DIR/COMPONENTS/poscar_atomic_type_periodic_table.py" "$experiment_dir/input/"
    # cp "$SCRIPT_DIR/COMPONENTS/poscar_atomic_type_one_hot.py" "$experiment_dir/input/"
    cp "$SCRIPT_DIR/COMPONENTS/space_group_one_hot.py" "$experiment_dir/input/"
}

# Step 5: Prepare Target Component
function prepare_target_component {
    local experiment_dir="$1"
    local target_file="$2"
    cp "$SCRIPT_DIR/COMPONENTS/$target_file" "$experiment_dir/output/"
}

# Step 6: Run the Experiment
function run_experiment {
    local experiment_dir="$1"
    local experiment_name="$2"
    conda run -n mmp python "$experiment_dir/NNN.py" $> "run_$experiment_name.txt"
}

# Step 7: Collect results
# Function to extract the highest validation accuracy
function extract_highest_validation_accuracy {
    local file="$1"
    grep 'Validation accuracy' "$file" | awk '
    {
        match($0, /Validation accuracy: \[([0-9.]+)\]/, arr)
        if (arr[1] > max_accuracy) {
            max_accuracy = arr[1]
            max_line = $0
        }
    }
    END {
        if (max_accuracy != "") {
            print max_line
        } else {
            print " - "
        }
    }
    '
}

# Function to process all result files
function process_results {
    for result_file in run_*.txt; do
        printf "${BLUE} $result_file: ${NC}"
        extract_highest_validation_accuracy "$result_file"
    done
}

# Main Execution
echo -e "${CYAN}Starting concurrent experiment setup and execution...${NC}"
install_dependencies

for target_file in "$SCRIPT_DIR/COMPONENTS/"*; do
    if [[ "$(basename "$target_file")" != poscar_* ]]; then
        experiment_dir="$SCRIPT_DIR/experiment_$(basename "$target_file" .py)"
        printf "\n\n\n\n\n  ${CYAN}Starting experiment: $(basename "$target_file")...${NC}\n"
        create_experiment_folder "$experiment_dir"
        extract_poscar "$experiment_dir"
        select_model "$experiment_dir"
        prepare_input_components "$experiment_dir"
        prepare_target_component "$experiment_dir" "$(basename "$target_file")"
        run_experiment "$experiment_dir" "$(basename "$target_file")"
    fi
done

echo -e "${GREEN}All experiments have been set up and are running concurrently.${NC}"
wait
echo -e "${GREEN}All experiments have completed.${NC}"
process_results
