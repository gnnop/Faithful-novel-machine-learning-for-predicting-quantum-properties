from includes import *
# Create an instance of the loader to keep the CSV loaded
csv_loader = CSVLoader('magnetic_ordering.csv')

def classifier():
    return loss_classification

# Functions to be used in another module
def valid_ids():
    return csv_loader.valid_ids()

def info(id):
    temp = csv_loader.info(id)
    arr = [0.0]*4
    if temp == "AFM":
        arr[0] = 1.0
    elif temp == "FiM":
        arr[1] = 1.0
    elif temp == "FM":
        arr[2] = 1.0
    elif temp == "NM":
        arr[3] = 1.0
    
    return arr