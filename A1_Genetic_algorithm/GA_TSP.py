
import numpy as np
import random
import json
import sys
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
# We need to import modules from the current
# and parent directories.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(current)
sys.path.append(parent)
import aux
from EA import Evolutive_algorithm

class Genetic_Algorithm_TSP(Evolutive_algorithm):
    """
    Child class from AE implementing a genetic algorithm (GA) for the Traveling
    Salesman Problem (TSP).
    """

    def __init__(self, name, instance, parameters_file=None, known_optimal=None):
        """
        Initialize the algorithm with the given parameters and cities in the 
        corresponding files.

        :parameters
        -instance_file: The path of a tsp file (a text file where every 
        row represents a city, with the format city_id x_coord y_coord).
         -parameters_file: The path of a json file containing the parameters 
        as a dictionary {key:value}. If None, the default parameters will
        be loaded.
        -known_optimal: The known optimal solution or upper bound for the 
        solution to the problem (int or tuple), if available.
        """

        # The instance name will be the name of the isnatcne file without the
        # extension.

        # Load the city coordinate matrix from the instance file
        self.cities = aux.load_instance(instance)
        self.n_cities = len(self.cities)
        self.city_distances = aux.calculate_city_distances(self.cities)
        # Load the specific GA parameters from the parameters file
        parameters = aux.load_parameters(parameters_file)
        # Number of generations
        self.n_gen = parameters["ngen"]
        # number of individuals of the population
        self.n_pop = parameters["npop"]
        # Selection probability
        self.ps = parameters["ps"]
        # Tournament size
        self.t_size = parameters["t_size"]
        # Numebr of torunaments
        self.n_tournaments = self.n_pop
        # Cross probability
        self.pc = parameters["pc"]
        # Mutation probability
        self.pm = parameters["pm"]
        # Elitism ()
        self.elitism = 1
        # Set the generic EA parameters
        self.n_children = self.n_pop

        super().__init__(name)  # Path(instance_file).stem
        self.parameters = parameters
        self.pool=Pool(processes=cpu_count())


    def init_pop(self):
        """
        Initialize the population matrix with random permutations of the city 
        indices.
        """
        print("Initializing population, calculating distance matrix...")

        city_idxs = np.arange(1, self.n_cities, dtype=int)
        return np.array([np.random.permutation(city_idxs)
                        for _ in range(self.n_pop)])

    def f_fit(self, x_0):
        """
        Calculate the fitness value of an individual or an array of individuals 
        based on the total distance of the route that they represent.
        x: individual or matrix of individuals (1d or 2d)
        """
        # Convert 1d input to a 2d row matrix
        x = np.atleast_2d(x_0)

        routes = np.concatenate((np.zeros((len(x), 1)),
                                 x,
                                 np.zeros((len(x), 1))),
                                axis=-1)

        # Get the pairwise distances between cities for each route
        city_start = routes[:, :-1].astype(int)
        city_end = routes[:, 1:].astype(int)

        L = self.city_distances[(city_start, city_end)]

        # Sum up the distances for each route
        length = np.sum(L, axis=1)

        # Return a scalar or vector of total lengths
        return length if x.ndim == 2 else length[0]

    def parent_selection(self):
        """
        Select parents for the next generation using tournaments.
        """
       
        
        parents_idx = [aux.tournament_selection(self.pop_fit, self.t_size)
                       for _ in range(self.n_tournaments)]
                                            
        return parents_idx

    def match_parents(self, parents):
        """ Match the parents 2 by 2 randomly and return a 3d array to index
        """
        # Since the parents are generated randomly,
        return np.array(parents,dtype=int).reshape(-1, 2)

    def select_survivors(self, children):
        """
        Substitute the previous popualtion with the children and aply elitism
        if applicable.
        """
        # Save the previous best individual for elitism and update the
        # population with the children
        prev_best_adapt = self.best_adapt
        prev_best = self.pop[self.best]
        # Update population
        self.pop[:] = children
        self.pop_fit[:] =  self.f_fit(self.pop)
        self.best = np.argmin(self.pop_fit)
        self.best_adapt = self.pop_fit[self.best]
        # If we apply elitism and the best individual of the previous generation
        # is better than the best individual of this generation
        if self.elitism and self.best_adapt > prev_best_adapt:
            # Subtitute the worst individual of this generation with the
            # best of the previous generation
            worst = np.argmax(self.pop_fit)
            self.pop[worst] = prev_best
            self.pop_fit[worst] = prev_best_adapt
            self.best_adapt = prev_best_adapt
            self.best = worst

    def variation_operators(self, parents):
        # Since the i-th parent will be selected with probability self.pc, 
        # instead of randomly egnerating one number every time we call the 
        # crossover function, we will make a mask to see which aprents will be 
        # selected. 
        crossover_mask = np.random.uniform(0, 1, len(parents))<self.pc
        
        result_cross=self.pool.map_async(aux.pmx_crossover, 
                                    parents[crossover_mask]).get()  # process data in batches
        
        parents[crossover_mask]=result_cross
        children=parents.reshape(-1,parents.shape[2])
        
        mutation_mask = np.random.uniform(0, 1, len(children))<self.pm
        result_mut=self.pool.map_async(aux.inter_mutation,
                                  children[mutation_mask]).get()
        children[mutation_mask]=result_mut

        return children
    
if __name__ == "__main__":

    instance = r"C:\Users\berti\Mi_equipo\Documentos\Roberto\PROGRAMMING\PYTHON\Comput_Evol\A1_Genetic_algorithm\instances\complex.tsp"
    parameters_file = r"C:\Users\berti\Mi_equipo\Documentos\Roberto\PROGRAMMING\PYTHON\Comput_Evol\A1_Genetic_algorithm\params\complex\pc_1.json"
    a = Genetic_Algorithm_TSP(
        name="Test", instance=instance, parameters_file=parameters_file)
    a.run()
