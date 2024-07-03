from includes import *
# Create an instance of the loader to keep the CSV loaded
csv_loader = CSVLoader('space_group.csv')

def classifier():
    return loss_classification

# Functions to be used in another module
def valid_ids():
    return csv_loader.valid_ids()

def info(id):
    temp = csv_loader.info(id)
    arr = [0] * 231
    arr[int(temp)] = 1
    return arr
