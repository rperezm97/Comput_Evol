
import numpy as np
import sys


import os
A1_FOLDER = os.path.dirname(__file__)
sys.path.append(A1_FOLDER)
import aux
sys.path.append( os.path.join(A1_FOLDER, ".."))
from EA import Evolutive_algorithm

sys.path.append("..")
from EA import Evolutive_algorithm

class Genetic_Algorithm(Evolutive_algorithm):
    """
    Child class from AE implementing a genetic algorithm (GA) for a fucntion 
    optimization problem.
    """
    # ----------------INITIALIZATION OF VARIABLES AND PARAMETERS---------------
    def __init__(self, instance_id):
        """ Initialize the GA class """
        print(f"\nInitializing the Genetic Algorithm\n")
        
        # Initiate the  base_folder it to load the instance, the
        # parameters, etc. 
        self.base_folder=os.path.dirname(__file__)
        # Call the aprent cosntructor
        super().__init__(instance_id)  
        
        #self.pool=Pool(processes=cpu_count())
    
    def load_instance(self):
        """Load the problem instance an initialize them as attributes in-place.
        This is loaded from aux to be problem-agnostic"""
        aux.load_instance(self)
        
    def init_pop(self):
        """
        Randomly initialize the (genotypic) population matrix with binary values.
        """
        print("Initializing population...")

        return np.random.randint(0, 2, 
                                size=(self.n_pop, 
                                      self.n_bits_per_dim*self.n_dimensions))
    
    
    # ------------------ METHODS FOR THE GA EVOLUTION -----------------------
    def f_fitness(self, x):
        """Evaluate the fitness function on an individual. The fitness function
        is implemented in aux to be problem-agnostic"""
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
        selected_parents = np.random.choice(np.arange(self.n_pop), 
                                            size=self.n_pop, 
                                            p=selection_p,
                                            replace=True)

        return selected_parents 
    
    def variation_operators(self, selected_parents):
        """
        Apply the variation operators of the GA, crossover and mutation, over 
        the selected parents 
        """
        # Since the parent selection is random, we don't need to shuffle 
        # between pairing. We just reshape it
        parent_matches = selected_parents.reshape(-1, 2)
        
        # Since the i-th parent will be selected with probability self.pc, 
        # instead of randomly egnerating one number every time we call the 
        # crossover function, we will make a mask to see which aprents will be 
        # selected. 
        
        matches_genome=self.pop[parent_matches]
        crossover_mask = np.random.uniform(0, 1, len(matches_genome))<self.pc
        
        # If the aprents are the same, then we don't ened to cross them, 
        # teh result would be teh same
        crossover_mask &= np.all(matches_genome[:, 0, :] != matches_genome[:, 1, :], 
                                 axis=1)
        
        if np.any(crossover_mask):
            result_cross=list(map(lambda pair : self.crossover(pair), 
                                                matches_genome[crossover_mask])) 
            matches_genome[crossover_mask]=result_cross
            
        children=matches_genome.reshape(-1,matches_genome.shape[2])

        return self.mutation(children)
    
    def crossover(self,parents):
        """
        Perform two-point crossover on the parents.
        """
        
        p1, p2 = parents
        
        a=b=0
        # The while loop is to avoid that the two cross points are the same
        while a==b:
            # The crossing oeprator must preserve the information of each 
            # dimensions, just interchange whole variables. For that, the 
            # crossing point must be a multiple of n_bits_per_dim
            a,b = np.sort(np.random.randint(0, self.n_dimensions, 
                                            size=2)) * self.n_bits_per_dim

        # Create two empty children
        c1 = p1.copy()
        c2 = p2.copy()

        # Perform the crossover
        c1[a:b]= p2[a:b]
        c2[a:b]= p1[a:b]
        return c1, c2

    def mutation(self, pop):
        """
        Perform a mutation on individual x by flipping each gene's binary value 
        with a probability of pm.
        """
        # Create a random mask with the same shape as x.
        # Each gene is mutated with a probability of pm
        random_mask = np.random.rand(*pop.shape) < self.pm

        # Perform XOR operation to mutate x
        # Since x is a boolean array, we can use the != operator, which acts as XOR for booleans
        mutated_pop = pop != random_mask

        return mutated_pop
    
    

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

    a = Genetic_Algorithm("Sphere_B1_test")
    a.evolve()
