# TODO LIST


- models should write results out for training automatically, and write their parameters out each epoch to avoid failed training
- if a model is run and there's a prediction.csv, the model should write out the material predictions to the csv using the existing param files and then turn off. This allows easy retrieval of practical results
- all models should use the method of validating with batch size currently incorporated in CANN and CGNN
- there should be a correspondence set up between the paper and the experiments, so you can run a file to get the list of labelled experiments that reproduce results in the paper
- 0.89 validation accuracy must be surpassed on TQC advanced for the paper to be credible. I acheived this once with CGNN, I am trying to do it again.
- all hyperparameters must be moved to the top and labeled in the code
- once 0.85 accuracy is reached with a model, it can be tested for generalization on the rest of the tasks in the paper
- once 0.85 accuracy is reached with a model, and training accuracy is at 0.99, all 4 models should be combined to determine what materials the models are collectively struggling with. These will be the 'of interest' materials highlighted in the paper
- some experiments need to be listed with justification for why results go one way or another
- Once this is all done and written up, we're ready for publication



# Materials Modeling Project

This project provides machine learning methods that are theoretically fully general, capable of modeling any material property desires. It is designed specifically to be simple to use on novel materials problems.

## Project Structure

- **COMPONENTS/**: Contains various components used in experiments.
- **CSV/**: Contains CSV files for data processing.
- **MODELS/**: Contains different models used for experiments.
- **POSCAR/**: Contains POSCAR files for material structures. Must be unzipped.

## Setup and Running Experiments

### Automated (Linux only)

An interactive experiment setup wizard is provided for Linux users. Run `bash wizard.bash` to get started. This script needs [`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) and [`7z`](https://www.7-zip.org/download.html)

### Manual (All operating systems)

To run an experiment, follow these steps:

0. **Install Dependencies**
   - Initialize a new virtual environment upon setting up this project.
   `conda env create -f environment.yaml`. This will install most dependencies necessary to train and apply the models.
   - Install JAX in that environment: `conda run -n mmp pip install jax chex optax dm-haiku jraph`.
      - Note: Installing the JAX libraries remains separate from installing the rest of the packages to give you flexibility in the way JAX is installed. Refer to [the official JAX installation details](https://jax.readthedocs.io/en/latest/installation.html) for more detailed information.
   - Every time this project is used in a new terminal, run `conda activate mmp` to activate this new environment before starting any experiments.

1. **Create a Folder for the Experiment**
   - Make a new directory for your experiment. It must be directly below the repository directory.

2. **Select a Model**
   - Copy a model file from the `MODELS` folder into your experiment directory.

3. **Create Subfolders**
   - Inside your experiment directory, create two subfolders:
     - `input/`
     - `output/`

4. **Prepare Input Components**
   - Copy the desired components into the `inputs/` folder.
   - Ensure to include at least one `poscar_atomic_` type component, as POSCAR files are always parsed.

5. **Prepare Target Component**
   - Copy components into the `target/` folder.
   - Components labeled `poscar_` are not valid in the target folder as POSCAR files are always assumed to be inputs. This is open to modification.

## Running the Models

Double click the model file or run it from the command line.

Models will attempt to load two pickle files if they exist:
- `model.params`: Contains model parameters.
- `processed.pickle`: Contains processed data.

> **Note:** If you want a clean slate to run from, these files must be deleted.

### Model Configuration

All models by default:
- Perform a 90-10 split for testing and training data.
- Have important and easily changed functions labeled at the top.
- Do not currently incorporate hyperparameter search.
- Are multithreaded. If your computer experiences a large slowdown during training, either decrease the thread number or the batch size, both located at the top in the models.py that you chose.

## Additional Information

- The `includes.py` file provides additional functionality and should be included in your experiment setup if needed.
- The `how to use.txt` file contains additional instructions and tips for using the project.

