from includes import *
# Create an instance of the loader to keep the CSV loaded
csv_loader = CSVLoader('energy_per_atom.csv')

def classifier():
    return loss_regression

# Functions to be used in another module
def valid_ids():
    return csv_loader.valid_ids()

def info(id):
    return getAbsoluteCoords(-1.0 * csv_loader.info(id))