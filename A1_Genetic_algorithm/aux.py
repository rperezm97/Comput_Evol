
import numpy as np
import os 
import json
import random

### LOADING DATA FUNCTIONS ###

def load_instance(instance):
    """
    Load the cities and the known optimum from the instance file specified 
    in instance_file.
    The file should be a text file where every row represents a city with 
    the format 'city_id x_coord y_coord'.

    :return: A numpy array with the city coordinates.
    """
    if os.path.isfile(instance):
        with open(instance) as fp:
            f = fp.readlines()
            cities = np.loadtxt(f, dtype=int)[:, 1:]
    else:
        raise ValueError("Select valid instance file")
    return cities

def load_parameters(parameters_file):
    """
    Load the parameters from the json file specified in param_file. 
    If param_file is None or the file is not found, default parameters will 
    be used.

    :return: A dictionary containing the parameters.
    """
    try:
        print("Loading parameters from file.\n")
        with open(parameters_file) as fp:
            parameters = json.load(fp)
    except:
        print("Invalid/empty parameter file.\nUsing default parameters.\n")
        parameters = {}
        parameters["n_gen"] = 100
        parameters["n_pop"] = 10
        parameters["ps"] = 1
        parameters["t_size"] = 3
        parameters["n_tournaments"] = parameters["n_pop"]
        parameters["pc"] = 1
        parameters["pm"] = 1
        parameters["elitism"]=1
        # save_parameters(parameters, "parmeters/default.json")
    return parameters

def save_parameters(parameters,parameters_file):
    """
    Save the parameters to the json file specified in param_file.
    """
    with open(parameters_file, "w") as fp:
        json.dump(parameters, fp)

def modify_parameters(parameters, key, value):
    """
    Modify a parameter and save it to the parameter file.

    :param key: The parameter to be modified.
    :param value: The new value for the parameter.
    """
    parameters[key] = value
    save_parameters()


### HELPER FUNCTIONS FOR GA OPERATORS ###

def calculate_city_distances(cities):
    """
    Calculate a matrix with the pairwise distances between the cities.
    """
    ncities = len(cities)
    city_distances = np.empty((ncities, ncities))
    
    # Set 0s to diagonal elements
    city_distances[tuple((range(ncities), range(ncities)))] = 0

    # Calculate the indices of the lower triangular matrix:
    # pairs of city indices (i1, i2) where i1 < i2
    tri_idx = np.tril_indices(n=ncities, k=-1)

    # Create an array of tuples with the pairs of cities
    pairs = cities[np.array(tri_idx)]

    # Calculate the pairwise distances
    d = np.sqrt(np.sum((pairs[0]-pairs[1])**2, axis=-1))

    # Assign the distances to the lower and upper triangular parts of the matrix
    city_distances[tri_idx] = d
    permut_idx = np.roll(np.array(tri_idx), 1, axis=0)
    city_distances[tuple(permut_idx)] = d

    return city_distances

def tournament_selection(pop_fit,t_size):
        """
        Select a parent from the population using tournament selection.
        """
        n_pop=len(pop_fit)
        indiv_idx = random.sample(range(n_pop), k=t_size)
        # TODO: What if theres more than one min?
        idx = min(indiv_idx, key=lambda i: pop_fit[i]) 
        return idx

def find_new_pos(p, c):
    """
    Find the positions of the missing genes in parent 'p' that are not in 
    child 'c' and add them to 'c'.

    Parameters:
        p (numpy.ndarray): Parent from the partially mapped crossover.
        c (numpy.ndarray): Child from the partially mapped crossover.

    Returns:
        numpy.ndarray: Child with missing genes added.

    """
    # Find the indices of the free positions in child 'c'
    free = np.where(c == 0)[0]
    # Find the indices of the remaining genes in parent 'p' that are not in 
    # child 'c'
    rest_id = np.where((1-np.in1d(p, c)))
    # Loop over the remaining genes in parent 'p'
    for i in p[rest_id]:
        # Find the index of the gene in parent 'p'
        pos = np.where(p == i)[0][0]
        # While the index is not free in child 'c', continue looking for a free
        # position
        
        while pos not in free:
            k = c[pos]
            pos = np.where(p == k)[0][0]
        # Add the gene to the first free position in child 'c'
        c[free[0]] = i
        # Update the list of free positions in child 'c'
        free = free[1:]
    
    return c
