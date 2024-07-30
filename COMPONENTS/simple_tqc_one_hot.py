from includes import *

# create an instance of the loader to keep the csv loaded
csv_loader = CSVLoader("simple_tqc.csv")

# functions to be used in another module


def valid_ids():
    return csv_loader.valid_ids()


def classifier():
    return EvaluationMethods.classification


def info(id):
    temp = csv_loader.info(id)
    arr = [0.0] * 3
    if temp == "trivial":
        arr[0] = 1.0
    elif temp == "TI":
        arr[1] = 1.0
    elif temp == "SM":
        arr[2] = 1.0

    return arr
