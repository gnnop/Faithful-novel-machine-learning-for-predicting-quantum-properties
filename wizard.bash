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

# ASCII Art Banner
function print_banner {
    echo -e "${BLUE}"
    cat << "EOF"
╔╦╗┌─┐┌┬┐┌─┐┬─┐┬┌─┐┬  ┌─┐  ╔╦╗┌─┐┌┬┐┌─┐┬  ┬  ┬┌┐┌┌─┐  ╔═╗┬─┐┌─┐ ┬┌─┐┌─┐┌┬┐
║║║├─┤ │ ├┤ ├┬┘│├─┤│  └─┐  ║║║│ │ ││├┤ │  │  │││││ ┬  ╠═╝├┬┘│ │ │├┤ │   │ 
╩ ╩┴ ┴ ┴ └─┘┴└─┴┴ ┴┴─┘└─┘  ╩ ╩└─┘─┴┘└─┘┴─┘┴─┘┴┘└┘└─┘  ╩  ┴└─└─┘└┘└─┘└─┘ ┴ 
          ╔═╗┌─┐┌┐┌┌─┐┬┌─┐┬ ┬┬─┐┌─┐┌┬┐┬┌─┐┌┐┌  ╦ ╦┬┌─┐┌─┐┬─┐┌┬┐           
          ║  │ ││││├┤ ││ ┬│ │├┬┘├─┤ │ ││ ││││  ║║║│┌─┘├─┤├┬┘ ││           
          ╚═╝└─┘┘└┘└  ┴└─┘└─┘┴└─┴ ┴ ┴ ┴└─┘┘└┘  ╚╩╝┴└─┘┴ ┴┴└──┴┘           
EOF
    echo -e "${NC}"
}

# Function to detect NVIDIA driver version and suggest CUDA version
function detect_cuda_version {
    driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
    if [[ -z "$driver_version" ]]; then
        echo -e "${YELLOW}No NVIDIA driver detected. Defaulting to CPU version of JAX.${NC}"
        cuda_version=""
    else
        echo -e "${GREEN}Detected NVIDIA driver version: $driver_version${NC}"

        # Parse major and minor driver version
        IFS='.' read -ra driver_parts <<< "$driver_version"
        driver_major=${driver_parts[0]}
        driver_minor=${driver_parts[1]}

        # Determine the appropriate CUDA version
        if ((driver_major >= 450)); then
            echo -e "${GREEN}CUDA 12 is supported.${NC}"
            cuda_version="cuda12"
        else
            echo -e "${RED}Unsupported NVIDIA driver version. Please upgrade your driver to at least version 450.${NC}"
            exit 1
        fi
    fi
}

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
        detect_cuda_version
        if [[ -z "$cuda_version" ]]; then
            conda run -n mmp pip install jax
        else
            conda run -n mmp pip install jax jaxlib==0.4.14+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_releases.html
        fi
    fi
}

# Step 1: Create a Folder for the Experiment
function create_experiment_folder {
    while true; do
        echo -e "${CYAN}Enter the name of your experiment directory: ${NC}"
        read experiment_dir
        if [[ -z "$experiment_dir" ]]; then
            echo -e "${RED}Experiment directory name cannot be empty. Please enter a valid name.${NC}"
        elif [[ -d "$experiment_dir" ]]; then
            echo -e "${RED}Experiment directory already exists. Do you want to overwrite it? (y/n): ${NC}"
            read overwrite
            if [[ "$overwrite" == "y" ]]; then
                rm -rf "$experiment_dir"
                mkdir -p "$experiment_dir/input" "$experiment_dir/output"
                echo -e "${GREEN}Experiment directory '$experiment_dir' created with 'inputs' and 'target' subfolders.${NC}"
                break
            elif [[ "$overwrite" == "n" ]]; then
                while true; do
                    echo -e "${CYAN}Do you want to run the existing experiment now? (y/n): ${NC}"
                    read run_existing
                    if [[ "$run_existing" == "y" ]]; then
                        echo -e "${YELLOW}Running the existing experiment...${NC}"
                        model_file=$(find "$experiment_dir" -maxdepth 1 -name "*.py" -not -name "includes.py" -print -quit)
                        if [[ -n "$model_file" ]]; then
                            conda run -n mmp python "$model_file"
                        else
                            echo -e "${RED}No model script found in the existing experiment directory.${NC}"
                        fi
                        exit 0
                    elif [[ "$run_existing" == "n" ]]; then
                        echo -e "${YELLOW}Please enter a different directory name.${NC}"
                        break
                    else
                        echo -e "${RED}Invalid response. Please enter 'y' for yes or 'n' for no.${NC}"
                    fi
                done
            else
                echo -e "${RED}Invalid response. Please enter 'y' for yes or 'n' for no.${NC}"
            fi
        else
            mkdir -p "$experiment_dir/input" "$experiment_dir/output"
            echo -e "${GREEN}Experiment directory '$experiment_dir' created with 'inputs' and 'target' subfolders.${NC}"
            break
        fi
    done
    cp "$SCRIPT_DIR/includes.py" "$experiment_dir"
}

# Step 2: Select a Model
function select_model {
    echo -e "${BLUE}Available models:${NC}"
    ls "$SCRIPT_DIR/MODELS"
    while true; do
        echo -e "${CYAN}Enter the model filename you want to copy to your experiment directory: ${NC}"
        read model_file
        if [[ -f "$SCRIPT_DIR/MODELS/$model_file" ]]; then
            cp "$SCRIPT_DIR/MODELS/$model_file" "$experiment_dir/"
            echo -e "${GREEN}Model '$model_file' copied to '$experiment_dir'.${NC}"
            break
        else
            echo -e "${RED}Model file does not exist. Please enter a valid filename from the list above.${NC}"
        fi
    done
}

# Step 3: Extract POSCAR.7z if POSCAR directory is missing
function extract_poscar {
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
    echo -e "${BLUE}Available components:${NC}"
    ls "$SCRIPT_DIR/COMPONENTS"
    echo -e "${YELLOW}Copying selected components to the 'inputs' folder...${NC}"
    poscar_atomic_included=false
    while true; do
        echo -e "${CYAN}Enter the component filename to copy (or 'done' to finish): ${NC}"
        read component_file
        if [[ "$component_file" == "done" ]]; then
            if [ "$poscar_atomic_included" = false ]; then
                echo -e "${RED}At least one poscar_atomic_ type component is required.${NC}"
            else
                break
            fi
        elif [[ -f "$SCRIPT_DIR/COMPONENTS/$component_file" ]]; then
            cp "$SCRIPT_DIR/COMPONENTS/$component_file" "$experiment_dir/input/"
            echo -e "${GREEN}Component '$component_file' copied to '$experiment_dir/input/'.${NC}"
            if [[ "$component_file" == poscar_atomic_* ]]; then
                poscar_atomic_included=true
            fi
        else
            echo -e "${RED}Component file does not exist. Please enter a valid filename from the list above.${NC}"
        fi
    done
}

# Step 5: Prepare Target Component
function prepare_target_component {
    echo -e "${BLUE}Available components:${NC}"
    ls "$SCRIPT_DIR/COMPONENTS"
    while true; do
        echo -e "${CYAN}Enter the target component filename to copy: ${NC}"
        read target_file
        if [[ "$target_file" != poscar_* ]]; then
            if [[ -f "$SCRIPT_DIR/COMPONENTS/$target_file" ]]; then
                cp "$SCRIPT_DIR/COMPONENTS/$target_file" "$experiment_dir/output/"
                echo -e "${GREEN}Target component '$target_file' copied to '$experiment_dir/output/'.${NC}"
                break
            else
                echo -e "${RED}Target component file does not exist. Please enter a valid filename from the list above.${NC}"
            fi
        else
            echo -e "${RED}Invalid target component. POSCAR files are not valid in the target folder. Please enter a different filename.${NC}"
        fi
    done
}

# Step 6: Run the Experiment
function run_experiment {
    while true; do
        echo -e "${CYAN}Do you want to run the experiment now? (y/n): ${NC}"
        read run_now
        if [[ "$run_now" == "y" ]]; then
            echo -e "${YELLOW}Running the experiment...${NC}"
            # Assuming the model script to run is the same as the selected model file
            conda run -n mmp python "$experiment_dir/$model_file"
            break
        elif [[ "$run_now" == "n" ]]; then
            echo -e "${GREEN}Setup completed. You can run your experiment from the '$experiment_dir' directory.${NC}"
            break
        else
            echo -e "${RED}Invalid response. Please enter 'y' for yes or 'n' for no.${NC}"
        fi
    done
}

# Main Execution
print_banner

echo -e "${CYAN}Welcome to the Auto Setup Wizard for the Materials Modeling Project!${NC}"


echo -e "${BLUE}"
cat << "EOF"

(0/6)
╦┌┐┌┌─┐┌┬┐┌─┐┬  ┬    ╔╦╗┌─┐┌─┐┌─┐┌┐┌┌┬┐┌─┐┌┐┌┌─┐┬┌─┐┌─┐
║│││└─┐ │ ├─┤│  │     ║║├┤ ├─┘├┤ │││ ││├┤ ││││  │├┤ └─┐
╩┘└┘└─┘ ┴ ┴ ┴┴─┘┴─┘  ═╩╝└─┘┴  └─┘┘└┘─┴┘└─┘┘└┘└─┘┴└─┘└─┘
EOF
echo -e "${NC}"
install_dependencies

echo -e "${BLUE}"
cat << "EOF"

(1/6)
╔═╗─┐ ┬┌─┐┌─┐┬─┐┬┌┬┐┌─┐┌┐┌┌┬┐  ╔═╗┌─┐┬  ┌┬┐┌─┐┬─┐
║╣ ┌┴┬┘├─┘├┤ ├┬┘││││├┤ │││ │   ╠╣ │ ││   ││├┤ ├┬┘
╚═╝┴ └─┴  └─┘┴└─┴┴ ┴└─┘┘└┘ ┴   ╚  └─┘┴─┘─┴┘└─┘┴└─
EOF
echo -e "${NC}"
create_experiment_folder
extract_poscar

echo -e "${BLUE}"
cat << "EOF"

(2/6)
╔═╗┌─┐┬  ┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌┬┐┌─┐┬  
╚═╗├┤ │  ├┤ │   │   ║║║│ │ ││├┤ │  
╚═╝└─┘┴─┘└─┘└─┘ ┴   ╩ ╩└─┘─┴┘└─┘┴─┘
EOF
echo -e "${NC}"
select_model

echo -e "${BLUE}"
cat << "EOF"

(3/6)
╔═╗┌─┐┬  ┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌┬┐┌─┐┬    ╦┌┐┌┌─┐┬ ┬┌┬┐┌─┐
╚═╗├┤ │  ├┤ │   │   ║║║│ │ ││├┤ │    ║│││├─┘│ │ │ └─┐
╚═╝└─┘┴─┘└─┘└─┘ ┴   ╩ ╩└─┘─┴┘└─┘┴─┘  ╩┘└┘┴  └─┘ ┴ └─┘
EOF
echo -e "${NC}"
prepare_input_components

echo -e "${BLUE}"
cat << "EOF"

(4/6)
╔═╗┌─┐┬  ┌─┐┌─┐┌┬┐  ╔╦╗┌─┐┌┬┐┌─┐┬    ╔╦╗┌─┐┬─┐┌─┐┌─┐┌┬┐
╚═╗├┤ │  ├┤ │   │   ║║║│ │ ││├┤ │     ║ ├─┤├┬┘│ ┬├┤  │ 
╚═╝└─┘┴─┘└─┘└─┘ ┴   ╩ ╩└─┘─┴┘└─┘┴─┘   ╩ ┴ ┴┴└─└─┘└─┘ ┴ 
EOF
echo -e "${NC}"
prepare_target_component

echo -e "${BLUE}"
cat << "EOF"

(5/6)
╔╦╗┌─┐┌┐┌┌─┐
 ║║│ ││││├┤ 
═╩╝└─┘┘└┘└─┘
EOF
echo -e "${NC}"
run_experiment

echo -e "${GREEN}All steps completed successfully.${NC}"

# End of script
