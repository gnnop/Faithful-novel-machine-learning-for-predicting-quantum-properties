from includes import *
# Create an instance of the loader to keep the CSV loaded
csv_loader = CSVLoader('advanced_tqc.csv')

# Functions to be used in another module
def valid_ids():
    return csv_loader.valid_ids()

def classifier():
    return loss_classification

def info(id):
    temp = csv_loader.info(id)
    arr = [0.0]*3
    if temp == "ES":
        arr[0] = 1.0#TSM
    elif temp == "ESFD":
        arr[0] = 1.0#TSM
    elif temp == "NLC":
        arr[1] = 1.0#TI
    elif temp == "SEBR":
        arr[1] = 1.0#TI
    elif temp == "LCEBR":
        arr[2] = 1.0#trivial

    return arr
