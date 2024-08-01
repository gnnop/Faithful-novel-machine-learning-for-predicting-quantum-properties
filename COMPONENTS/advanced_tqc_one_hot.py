from includes import *

# create an instance of the loader to keep the csv loaded
csv_loader = CSVLoader("tqc.csv")


def classifier():
    return EvaluationMethods.classification


# functions to be used in another module


def valid_ids():
    return csv_loader.valid_ids()


def info(id):
    temp = csv_loader.info(id)
    arr = [0.0] * 5
    if temp == "ES":
        arr[0] = 1.0
    elif temp == "ESFD":
        arr[1] = 1.0
    elif temp == "NLC":
        arr[2] = 1.0
    elif temp == "SEBR":
        arr[3] = 1.0
    elif temp == "LCEBR":
        arr[4] = 1.0

    return arr
