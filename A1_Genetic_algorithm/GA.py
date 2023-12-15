
import numpy as np
import sys
import os
import json
import time
from multiprocessing import Pool, cpu_count
# We need to import modules from the current
# and parent directories.
A1_FOLDER = os.path.dirname(__file__)
sys.path.append(A1_FOLDER)
import aux
import funct
sys.path.append( os.path.join(A1_FOLDER, ".."))
from EA import Evolutive_algorithm

class Genetic_Algorithm(Evolutive_algorithm):
    """
    Child class from AE implementing a genetic algorithm (GA) for a fucntion 
    optimization problem.
    """

    def __init__(self, instance_id):
        
        self.instance_id= instance_id
       
        aux.load_instance(self)
        
        # The instance idnetifier is  of the isnatcne file without the
        # extension.
        print(f"\nInitializing the Genetic Algorithm\n")
        
    
        #self.base_folder=os.path.dirname(__file__)
        super().__init__(instance_id, A1_FOLDER, self.optimal)  
        # Path(instance_file).stem
        
        
        #self.pool=Pool(processes=cpu_count())

    def init_pop(self):
        """
        Randomly initialize the (genotypic) population matrix with binary values.
        """
        print("Initializing population...")

        return np.random.randint(0, 2, 
                                size=(self.n_pop, 
                                      self.n_bits_per_dim*self.n_dimensions))
       
    def f_fitness(self, x):
        """Evaluate the fitness function on an individual"""
        return aux.f_fitness(x, self)
    
    def parent_selection(self):
            """
            Select the parents of teh current generation from 
            the population using ordering (linear) selection 
            """
        
            # Rank individuals based on their fitness values.
            # Higher fitness -> higher rank.
            sorted_indices = np.argsort(-self.pop_fitness)
            # Calculate selection probabilities.
            ranks= np.argsort(sorted_indices)
            
            selection_p =( ( (2-self.sel_pressure) / self.n_pop )
                           + (2 * (ranks) * (self.sel_pressure-1)
                           / (self.n_pop * (self.n_pop-1)))
                          ) 
            # Select parents with repetition
            parent_indices = np.random.choice(np.arange(self.n_pop), 
                                                size=self.n_pop, 
                                                p=selection_p,
                                                replace=True)

            return parent_indices 
    
    def variation_operators(self, parent_indices):
        
        # Since the parent selection is random, we don't need to shuffle 
        # between pairing. We just reshape it
        parent_matches = parent_indices.reshape(-1, 2)
        
        # Since the i-th parent will be selected with probability self.pc, 
        # instead of randomly egnerating one number every time we call the 
        # crossover function, we will make a mask to see which aprents will be 
        # selected. 
        
        parents=self.pop[parent_matches]
        crossover_mask = np.random.uniform(0, 1, len(parents))<self.pc
        
        # If the aprents are the same, then we don't ened to cross them, 
        # teh result would be teh same
        crossover_mask &= np.all(parents[:, 0, :] != parents[:, 1, :], axis=1)
        
        if np.any(crossover_mask):
            result_cross=list(map(lambda pair : aux.two_points_crossover(pair, 
                                                              self.n_dimensions), 
                                        parents[crossover_mask])) # process data in batches
            parents[crossover_mask]=result_cross
            
        children=parents.reshape(-1,parents.shape[2])

        return aux.bitflip_mutation(children, pm=self.pm)

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
        self.pop_fitness[:] =  self.f_fitness(self.pop)
        self.best = np.argmin(self.pop_fitness)
        self.best_adapt = self.pop_fitness[self.best]
        # If we apply elitism and the best individual of the previous generation
        # is better than the best individual of this generation
        if self.best_adapt > prev_best_adapt:
            # Subtitute the worst individual of this generation with the
            # best of the previous generation 
            worst = np.argmax(self.pop_fitness) 
            self.pop[worst] = prev_best
            self.pop_fitness[worst] = prev_best_adapt
            self.best_adapt = prev_best_adapt
            self.best = worst
    



if __name__ == "__main__":

    a = Genetic_Algorithm("SphereB1_test")
    a.run(early_stop=True)
