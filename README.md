# Materials Modeling Project

This project provides machine learning methods that are theoretically fully general, capable of modeling any material property desires. It is designed specifically to be simple to use on novel materials problems.

## Project Structure

- **COMPONENTS/**: Contains various components used in experiments.
- **CSV/**: Contains CSV files for data processing.
- **MODELS/**: Contains different models used for experiments.
- **POSCAR/**: Contains POSCAR files for material structures.

## Setup and Running Experiments

To run an experiment, follow these steps:

0. **Install Dependencies**
   - Initialize a new virtual environment upon setting up this project.
   `conda create --name mmp --file requirements.txt`. This will install all dependencies necessary to train and apply the models.

   - Before using this project in a new terminal, run `conda activate mmp` to activate this new environment.

1. **Create a Folder for the Experiment**
   - Make a new directory for your experiment.

2. **Select a Model**
   - Copy a model file from the `MODELS` folder into your experiment directory.

3. **Create Subfolders**
   - Inside your experiment directory, create two subfolders:
     - `inputs/`
     - `target/`

4. **Prepare Input Components**
   - Copy the desired components into the `inputs/` folder.
   - Ensure to include at least one `poscar_atomic_` type component, as POSCAR files are always parsed.

5. **Prepare Target Component**
   - Copy a single component into the `target/` folder.
   - Components labeled `poscar_` are not valid in the target folder as POSCAR files are always assumed to be inputs. This is open to modification.

## Running the Models

Models will attempt to load two pickle files if they exist:
- `model.params`: Contains model parameters.
- `processed.pickle`: Contains processed data.

> **Note:** If you want a clean slate to run from, these files must be deleted.

### Model Configuration

All models by default:
- Perform a 90-10 split for testing and training data.
- Have important and easily changed functions labeled at the top.
- Do not currently incorporate hyperparameter search.

## Usage

To run an experiment:

1. Set up your experiment directory as described above.
2. Run the model script from your experiment directory.

## Example

An example setup can be found in the `experiment_naive` folder.

## Additional Information

- The `includes.py` file provides additional functionality and should be included in your experiment setup if needed.
- The `how to use.txt` file contains additional instructions and tips for using the project.

