import numpy as np
import random
import json
import sys
import os
from pathlib import Path
# We need to import modules from the current
# and parent directories.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
print(current)
sys.path.append(current)
sys.path.append(parent)

from EA import Evolutive_algorithm
import aux

class Genetic_Algorithm_TSP(Evolutive_algorithm):
    """
    Child class from AE implementing a genetic algorithm (GA) for the Traveling
    Salesman Problem (TSP).
    """

    def __init__(self, name, instance_file, parameters_file=None, known_optimal=None):
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
        self.cities = aux.load_instance(instance_file)
        self.n_cities = len(self.cities)
        self.city_distances= aux.calculate_city_distances(self.cities)
        # Load the specific GA parameters from the parameters file
        parameters = aux.load_parameters(parameters_file)
        # Number of generations
        self.n_gen = parameters["n_gen"]
        # number of individuals of the population
        self.n_pop = parameters["n_pop"]
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
        self.elitism=parameters["elitism"]
        #Set the generic EA parameters
        self.n_children =self.n_pop
        
        super().__init__(name) #Path(instance_file).stem
        self.parameters=parameters
        
    def init_pop(self):
        """
        Initialize the population matrix with random permutations of the city 
        indices.
        """
        print("Initializing population, calculating distance matrix...")
        
        city_idxs = np.arange(1, self.n_cities)
        return np.array([np.random.permutation(city_idxs) 
                        for _ in range(self.n_pop)])

    def f_adapt(self, x):
        """
        Calculate the adaptation of an individual or an array of individuals 
        based on the total distance of the route that they represent.
        x: individual or matrix of individuals (1d or 2d)
        """
        # Convert 1d input to a 2d row matrix
        x = np.atleast_2d(x)

        routes = np.concatenate((np.zeros((len(x), 1)), 
                                 x, 
                                 np.zeros((len(x), 1))),
                                 axis=-1)

        # Get the pairwise distances between cities for each route
        city_start = routes[:,:-1].astype(int)
        city_end = routes[:,1:].astype(int)
        
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
        if self.n_pop % 2:
            parents_idx.append(None)
        return np.array(parents_idx)

    def match_parents(self, parents):
        """ Match the parents 2 by 2 randomly and return a 3d array to index
        """
        # Since the parents are generated randomly,  
        return parents.reshape(-1, 2)
    
    def crossover(self, parents):
        """
        Perform partially mapped crossover on the parents.
        """
        p1, p2 = parents
        # If crossover probability is not met or second parent is None, 
        # return parents
      
        if (random.random() >= self.pc or p2 is None):
            return p1, p2
        
        #print("cross",p1,p2)
        n = len(p1)
        # Choose two random crossover points
        b = np.random.randint(0, n-1)
        a = np.random.randint(0, n-1)
        start = min(a, b)
        end = max(a, b)

        # Create two empty children
        c1 = np.zeros(n, dtype=float)
        c2 = np.zeros(n, dtype=float)

        # Copy the segment from parent 1 to child 1
        c1[start:end+1] = p1[start:end+1]
        # Copy the segment from parent 2 to child 2
        c2[start:end+1] = p2[start:end+1]

        # Get the complement of the segment from parent 1 in child 2
        notc1 = p1-c1
        # Get the complement of the segment from parent 2 in child 1
        notc2 = p2-c2

        # Add the missing genes to child 1 and child 2
        notc1_in_notc2 = np.in1d(notc2, notc1).astype(float)
        c1+= notc2*notc1_in_notc2
        notc2_in_notc1 = np.in1d(notc1, notc2)
        c2 += notc1*notc2_in_notc1

        # Find the remaining missing genes and add them to child 1 and child 2
        c1 = aux.find_new_pos(p1, c1)
        c2 = aux.find_new_pos(p2, c2)

        #print("cross",a,b,c1,c2)
        return c1, c2
    
    
    def mutate(self, x):
        """
        Perform a mutation on individual x by swapping two randomly chosen genes.
        """
         # If the random number generated is greater than the mutation probability,
        # return the original individual
        if random.random() >= self.pm:
            
            return x
        # Get the length of the individual
        n = len(x)
        # Randomly select two genes to be swapped
        i = np.random.randint(0, n-1)
        j = np.random.randint(0, n-1)
        # Make sure i != j to introduce variation
        while j == i:
            j = np.random.randint(0, n-1)
        # Swap the two selected genes
        x[i], x[j] = x[j], x[i]
        return x

    def select_survivors(self, children):
        """
        Substitute the previous popualtion with the children and aply elitism
        if applicable.
        """
        # Save the previous best individual for elitism and update the
        # population with the children
        prev_best_adapt=self.best_adapt
        prev_best=self.pop[self.best]
        self.pop=children
        self.pop_fit = self.f_adapt(children)
        self.best_adapt = np.min(self.pop_fit)
        self.best = np.argmin(self.pop_fit)
        # If we apply elitism and the best individual of the previous generation
        # is better than the best individual of this generation
        if self.elitism and  self.best_adapt>prev_best_adapt:
            # Subtitute the worst individual of this generation with the 
            # best of the previous generation
            worst=np.argmax(self.pop_fit)
            self.pop[worst]=prev_best
            self.pop_fit[worst]=prev_best_adapt
            self.best_adapt=prev_best_adapt

if __name__=="__main__":
    instance="/root/PYTHON/Comput_Evol/A1_Genetic_algorithm/instances/simple.tsp"
    a=Genetic_Algorithm_TSP(name="Test", instance_file=instance)
    a.run()