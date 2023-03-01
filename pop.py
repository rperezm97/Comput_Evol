import numpy as np
import random
from functools import partial
from multiprocessing import Pool

class Evolutive_algoritm:
    """Abstract class that implements the methods of a generic evolutionary
    algorithm and serves as a parent for other algorithms."""
    
    def __init__(self):
        """Initialize the population, calculate the adaptation of each
        individual, and store the best individual and its adaptation."""
        self.pop = self.init_pop()
        self.pop_fit= self.f_adapt(self.pop)
        self.best_adapt = np.min(self.pop_fit)
        self.best = np.argmin(self.pop_fit)
        
        self.n_pop = None
        self.num_parents=None 
        self.num_children=None
    def init_pop(self):
        """Initialize the population. To be implemented by child classes."""
        pass
        
    def f_adapt(self, pop):
        """Calculate the adaptation of each individual in the population.
        To be implemented by child classes."""
        pass

    def parent_selection(self):
        """Select parents from the population. To be implemented by child 
        classes."""
        pass
    
    def crossover(self, parents):
        """Perform crossover on two parents to create two children. To be
        implemented by child classes."""
        pass

    def mutate(self, individual):
        """Perform mutation on an individual. To be implemented by 
        child classes."""
        pass
    
    def select_survivors(self, parents, children):
        """Select the individuals to be included in the next generation.
        To be implemented by child classes."""
        pass
    def reproduce(self):
        """Select parents, perform crossover and mutation, and update 
        the population and best individual."""
        # Select parents
        parents_idx = self.parent_selection()
        children = np.empty((self.num_children, self.pop.shape[1]))

        # Reshape the parent indices array into a 3D array
        parent_indices = parents_idx.reshape(-1, 
                                             self.num_parents)

        # Apply crossover and mutation to the entire array of parent indices
        children = self.crossover(self.pop[parent_indices])
        children = self.mutate(children)

        # Update the population and best individual
        self.pop = self.select_survivors(children)
        self.pop_fit= self.f_adapt(self.pop)
        self.best_adapt = np.min(self.pop_fit)
        self.best = np.argmin(self.pop_fit)