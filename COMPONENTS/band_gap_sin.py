from includes import *

# create an instance of the loader to keep the csv loaded
csv_loader = CSVLoader("band_gap.csv")


def classifier():
    return EvaluationMethods.regression


# functions to be used in another module


def valid_ids():
    return csv_loader.valid_ids()


def info(id):
    return get_absolute_coords(float(csv_loader.info(id)), maxval=40)
