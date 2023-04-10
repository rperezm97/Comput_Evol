import numpy as np
import random
import json
import sys,os
# We need to import modules from the current
# and parent directories.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(current)
sys.path.append(parent)

from EA import Evolutive_algorithm
from aux import *

class GA_TSP(Evolutive_algorithm):
    """
    Child class implementing a genetic algorithm (GA) for the Traveling
    Salesman Problem (TSP).
    """

    def __init__(self, params_file, instance_file, known_optimal=None):
        """
        Initialize the algorithm with the given parameters and cities in the 
        corresponding files.
        
        :params
        -params_file: The path of a json file containing the parameters 
        as a dictionary {key:value}.
        -instance_file: The path of a tsp file (a text file where every 
        row represents a city, with the format city_id x_coord y_coord).
        -known_optimal: The known optimal solution or upper bound for the 
        solution to the problem (int or tuple), if available.
        """
        super().__init__()

        # Load the city matrix from the instance file
        self.instance_file = instance_file
        self.cities = self.load_instance()
        self.n_cities = len(self.cities)
        self.city_distances = self.calculate_city_distances(self.cities)

        # Load the parameters from the parameters file
        self.param_file = params_file
        params = self.load_params()
        self.npop = params["npop"]
        self.ps = params["ps"]
        self.t_size = params["t_size"]
        self.n_tournaments = self.npop
        self.pc = params["pc"]
        self.pm = params["pm"]

    ### LOADING DATA METHODS ###

    def load_instance(self):
        """
        Load the cities and the known optimum from the instance file specified 
        in self.instance_file.
        The file should be a text file where every row represents a city with 
        the format 'city_id x_coord y_coord'.

        :return: A numpy array with the city coordinates.
        """
        if os.path.isfile(self.instance_file):
            with open(self.instance_file) as fp:
                f = fp.readlines()
                cities = np.loadtxt(f, dtype=int)[:, 1:]
        else:
            raise ValueError("Select valid instance file")
        return cities

    def load_params(self):
        """
        Load the parameters from the json file specified in self.param_file. 
        If param_file is None or the file is not found, default parameters will 
        be used.

        :return: A dictionary containing the parameters.
        """
        try:
            print("Loading parameters from file.\n")
            with open(self.param_file) as fp:
                params = json.load(fp)
        except:
            print("Invalid/empty parameter file.\nUsing default parameters.\n")
            params = {}
            params["ngen"] = 100
            params["npop"] = 10
            params["ps"] = 1
            params["t_size"] = 2
            params["n_tournaments"] = params["npop"]
            params["pc"] = 0.2
            params["pm"] = 1
            self.params = params
            self.save_params()
        return params

    def save_params(self):
        """
        Save the parameters to the json file specified in self.param_file.
        """
        with open(self.param_file, "w") as fp:
            json.dump(self.params, fp)

    def modify_parameters(self, key, value):
        """
        Modify a parameter and save it to the parameter file.

        :param key: The parameter to be modified.
        :param value: The new value for the parameter.
        """
        self.parameters[key] = value
        self.save_params()
    
    ### GENETIC ALGORITHM METHODS ###
    def init_pop(self):
        """
        Initialize the population with random permutations of the city indices.
        """
        city_idxs = np.arange(1, self.n_cities)
        return [np.random.permutation(city_idxs) for i in range(self.npop)]

    def f_adapt(self, x):
        """
        Calculate the adaptation of the population based on the total distance 
        of the routes.
        """
        routes = np.vstack((np.append([0], x), np.append(x, [0])))
        L = self.city_distances[tuple(routes)]
        length = np.sum(L)
        return length

    def parent_selection(self):
        """
        Select parents for the next generation using tournaments.
        """
        parents_idx = [self.tournament_selection(self.pop_fit) for _ in range(self.n_tournaments)]
        if not self.npop % 2:
            parents_idx.append(None)
        return parents_idx

    def tournament_selection(self, pop_adapt):
        """
        Select a parent from the population using tournament selection.
        """
        indiv_idx = random.sample(range(self.npop), k=self.t_size)
        idx = min(indiv_idx, key=lambda i: pop_adapt[i])
        return idx

    def crossover(self, parents):
        """
        Perform partially mapped crossover on the parents.
        """
        p1, p2 = parents
        if (random.random() >= self.pc or not p1 or not p2):
            return p1, p2

        n = len(p1)
        # Choose two random crossover points
        b = np.random.randint(0, n-1)
        a = np.random.randint(0, n-1)
        start = min(a, b)
        end = max(a, b)

        # Create two empty children
        c1 = np.zeros(n, dtype=int)
        c2 = np.zeros(n, dtype=int)

        # Copy the segment from parent 1 to child 1
        c1[start:end+1] = p1[start:end+1]
        # Copy the segment from parent 2 to child 2
        c2[start:end+1] = p2[start:end+1]

        # Get the complement of the segment from parent 1 in child 2
        notc1 = p1-c1
        # Get the complement of the segment from parent 2 in child 1
        notc2 = p2-c2

        # Add the missing genes to child 1 and child 2
        notc1_in_notc2 = np.in1d(notc2, notc1)
        c1+= notc2*notc1_in_notc2
        notc2_in_notc1 = np.in1d(notc1, notc2)
        c2 += notc1*notc2_in_notc1

        # Find the remaining missing genes and add them to child 1 and child 2
        c1 = self.find_new_pos(p1, c1)
        c2 = self.find_new_pos(p2, c2)

        return c1, c2

    
    ### AUXILIARY METHODS
    def evolve(self):
        """
        Evolve the population for ngen generations and return the best individual
        and its fitness.
        """
        self.pop = self.init_pop()
        self.pop_fit = self.f_adapt(self.pop)
        best_idx = np.argmin(self.pop_fit)
        best = self.pop[best_idx]
        best_fit = self.pop_fit[best_idx]
        for i in range(self.ngen):
            parents_idx = self.parent_selection()
            parents = [self.pop[idx] for idx in parents_idx]
            children = [self.crossover(parents) for i in range(self.npop//2)]
            children = [self.mutate(child) for child in children]
            self.select_survivors(parents, children)
            best_gen_idx = np.argmin(self.pop_fit)
            if self.pop_fit[best_gen_idx] < best_fit:
                best = self.pop[best_gen_idx]
                best_fit = self.pop_fit[best_gen_idx]
                print(f"New best individual in generation {i+1}: {best}, \
                      with fitness {best_fit}.")
        return best, best_fit

    
    

    
