#os interop
import sys, os
import csv
from os.path import exists
import pickle

# Paths to the directories
input_dir = './input'
output_dir = './output'
#The csvs are called from input and output,so pop up two places. poscars is from the model, so only pop up one
poscars = os.path.abspath(os.path.join(os.path.dirname(__file__), '../POSCAR/'))
csvs = os.path.abspath(os.path.join(os.path.dirname(__file__), '../CSV/'))

import importlib.util

loss_classification = 0
loss_regression = 1




#ml libraries
import math
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import jraph


from scipy.spatial.transform import Rotation as R
import itertools
from dataclasses import dataclass
from multiprocessing import Pool, Manager
from typing import Any, Callable, Dict, List, Optional, Tuple
import functools
from random import shuffle
import re
from collections import Counter
import mendeleev


import time
import matplotlib.pyplot as plt


# Lists to hold the modules
poscar_globals = []
poscar_atomics = []
global_inputs = []
global_outputs = []

def load_module_from_file(file_path):
    """Load a module from a given file path and return the module object."""
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def process_directory(directory, is_input):
    """Process files in a given directory, loading modules based on filename prefixes."""
    for filename in os.listdir(directory):
        if filename.endswith('.py') and filename != 'includes.py':  # Only process python files
            file_path = os.path.join(directory, filename)
            if filename.startswith('poscar') and is_input:  # Only in input folder
                if filename.startswith('poscar_global'):
                    poscar_globals.append(load_module_from_file(file_path))
                elif filename.startswith('poscar_atomic'):
                    poscar_atomics.append(load_module_from_file(file_path))
            elif is_input:  # Other input files
                global_inputs.append(load_module_from_file(file_path))
            else:  # Output files
                global_outputs.append(load_module_from_file(file_path))

# Process each directory
def load_submodules():
    process_directory(input_dir, is_input=True)
    process_directory(output_dir, is_input=False)


#functions and utilities which may be shared but are included here to avoid repetition

#cgnn and ccnn - maybe csnn should incorporate?
completeTernary = list(itertools.product([-1, 0, 1], repeat=3))

#for poscar processing
def unpackLine(str):
    x = str.split()[0:3]
    return list(map(float, x))

def preprocessPoscar(id):
    poscar = open(poscars + "/" + str(id) + ".POSCAR", "r").read()
    inter = list(map(lambda a: a.strip(), poscar.split("\n")))
    return inter

def getSetOfPoscars():
    return set([x.split(".")[0] for x in os.listdir(poscars)])




#csv loader for components
class CSVLoader:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = {}
        self.load_csv()

    def load_csv(self):
        with open(csvs + "/" + self.file_name, mode='r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row:  # Ensure the row is not empty
                    self.data[row[0]] = row[1]

    def valid_ids(self):
        return set(self.data.keys())

    def info(self, id):
        return self.data.get(id, None)



class ExponentialDecayWeighting:
    def __init__(self, decay_rate=0.9):
        self.decay_rate = decay_rate
        self.accuracies = []
        self.weighted_sum = None
        self.normalization_factor = None

    def add_accuracy(self, accuracy):
        self.accuracies.append(accuracy)
        if len(self.accuracies) == 1:
            self.weighted_sum = accuracy
            self.normalization_factor = [1] * len(accuracy)
        else:
            self.weighted_sum = [self.weighted_sum[i] * self.decay_rate + accuracy[i] for i in range(len(accuracy))]
            self.normalization_factor = [self.normalization_factor[i] * self.decay_rate + 1 for i in range(len(accuracy))]

    def get_weighted_average(self):
        if not self.accuracies:
            return None
        return [self.weighted_sum[i] / self.normalization_factor[i] for i in range(len(self.weighted_sum))]

    def get_all_accuracies(self):
        return self.accuracies

#shared conversions
#Specifically, these functions are typically used by a couple files,
#but I pull them here due to how general they are. 
relativeCoordinateLength = 5
absoluteCoordinateLength = 2.0*5.0
fourierFeaturesPeriodScale = 2.0

def getRelativeCoordinates(val):
    '''
    This function is loosely inspired by the following paper:
        Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains
        by Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Jonathan T. Barron, and Ren Ng
    '''

    return [
        math.sin(
            math.tau
            * fourierFeaturesPeriodScale**(i+1)
            * val
        ) 
        for i in range(int(relativeCoordinateLength))
    ]

def getAbsoluteCoords(val):
    initialPeriod = 2**int(absoluteCoordinateLength/2)
    return [
        math.sin(
            math.tau
            * fourierFeaturesPeriodScale**(i+1)
            / initialPeriod
            *val
        ) 
        for i in range(int(relativeCoordinateLength / 2.0))
    ]

def flatten(nested_list):
    """Flattens a list of lists of arbitrary depth. yield was cool, but we need the whole list in mem anyway"""
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            # If item is a list, extend the flattened_list with the flattened items
            flattened_list.extend(flatten(item))
        else:
            # If item is not a list, append the item to flattened_list
            flattened_list.append(item)
    return flattened_list


#A function that returns the names of the elements for each one

def atom_names_in_poscar(poscar):
    atoms = poscar[5].split()
    numbs = poscar[6].split()

    total = 0
    for i in range(len(numbs)):
        total+=int(numbs[i])
        numbs[i] = total

    
    nodesArray = []

    curIndx = 0
    atomType = 0
    for i in range(total):
        curIndx+=1
        if curIndx > numbs[atomType]:
            atomType+=1
        
        nodesArray.append(atoms[atomType])
    
    return nodesArray

# Function to partition the dataset into training and validation sets
#example call:
#X_train, y_train, add_train, X_val, y_val, add_val = partition_dataset(0.4, data, labels, additional_data)
def partition_dataset(validation_percentage, *arrays):
    if not arrays:
        raise ValueError("At least one array must be provided")

    # Calculate the number of validation samples
    if isinstance(arrays[0], list):
        num_data = len(arrays[0])
    else:
        num_data = arrays[0].shape[0]
    num_val_samples = int(num_data * validation_percentage)

    # Generate shuffled indices
    indices = jnp.arange(num_data)
    shuffled_indices = jax.random.permutation(jax.random.PRNGKey(314), indices)


    val_indices = shuffled_indices[:num_val_samples]
    train_indices = shuffled_indices[num_val_samples:]
    
    # Split each array into training and validation sets
    train_sets = []
    val_sets = []

    for array in arrays:
        if isinstance(array, list):
            train_sets.append([array[i] for i in train_indices])
            val_sets.append([array[i] for i in val_indices])
        else:
            train_sets.append(array[train_indices])
            val_sets.append(array[val_indices])

    return (*train_sets, *val_sets)


#Consider, adding in more customizeability
def loss_fn(net, learning_type,learning_num, params, rng, inputs, targets):
    error = 0.0
    accuracy = jnp.array([])
    location = 0

    predictions = net.apply(params, rng, inputs,is_training=True)

    for i in range(len(learning_type)):
        if learning_type[i] == loss_regression:
            miss = jnp.mean((predictions[:,location:location + learning_num[i]] - targets[:,location:location + learning_num[i]]) ** 2)
            error += miss
            accuracy = jnp.append(accuracy, miss)
        elif learning_type[i] == loss_classification:
            error += jnp.sum(optax.softmax_cross_entropy( predictions[:,location:location + learning_num[i]], targets[:,location:location + learning_num[i]]))
            accuracy = jnp.append(accuracy, jnp.mean(jnp.argmax(predictions[:,location:location + learning_num[i]], axis=-1) == jnp.argmax(targets[:,location:location + learning_num[i]], axis=-1)))

        location += learning_num[i]
    
    return error, accuracy

#Only these functions get called on
def accuracy_fn(net, learning_type, learning_num, params, rng, inputs, targets):
    accuracy = jnp.array([])
    location = 0
    predictions = net.apply(params, rng, inputs,is_training=False)
    for i in range(len(learning_type)):
        if learning_type[i] == loss_regression:
            accuracy = jnp.append(accuracy, jnp.mean((predictions[:,location:location + learning_num[i]] - targets[:,location:location + learning_num[i]]) ** 2))
        elif learning_type[i] == loss_classification:
            accuracy = jnp.append(accuracy, jnp.mean(jnp.argmax(predictions[:,location:location + learning_num[i]], axis=-1) == jnp.argmax(targets[:,location:location + learning_num[i]], axis=-1)))
        location += learning_num[i]
    return accuracy
