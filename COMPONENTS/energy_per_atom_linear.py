from includes import *

# create an instance of the loader to keep the csv loaded
csv_loader = CSVLoader("energy_per_atom.csv")


def classifier():
    return EvaluationMethods.regression


# functions to be used in another module


def valid_ids():
    return csv_loader.valid_ids()


def info(id):
    v = csv_loader.info(id)
    if v is None:
        raise ValueError(
            f"energy_per_atom_linear: No entry in database exists for material {id}"
        )
    return [-1.0 * float(v)]
