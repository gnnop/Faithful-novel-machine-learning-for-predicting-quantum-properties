from includes import *

# create an instance of the loader to keep the csv loaded
csv_loader = CSVLoader("space_group.csv")


def classifier():
    return EvaluationMethods.classification


# functions to be used in another module


def valid_ids():
    return csv_loader.valid_ids()


def info(id):
    temp = csv_loader.info(id)
    return np.unpackbits(np.array([int(temp)], dtype=np.uint8)).tolist()
