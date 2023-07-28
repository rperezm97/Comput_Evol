import numpy as np
import random
from scipy.sparse import lil_matrix
import sys,os
#Local imports
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
print(current)
sys.path.append(current)
sys.path.append(parent)

from EA import Evolutive_algorithm
import aux
from kernel import KG,KP,KS
class GrammaticalEvolution(Evolutive_algorithm):
    """
    Class that represents the Grammatical Evolution algorithm to solve a linear
    regression problem.
    The goal is to approximate a function f, given pairs of points (x_j, f(x_j)), 
    using a combination
    of Gaussian, Polynomial and Sigmoid kernels.
    """

    def __init__(self, name, function, parameters_file=None, rules_file=None):
        """
        Initializes the Grammatical Evolution instance with the given filename 
        to load the parameters.
        """        
        #self.data = self.load_data(data_file)
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
        #
        self.n_samples=parameters["n_samples"]
        
        self.function=function
        self.sample=aux.get_sample(self.function, 
                                    self.n_samples)
        #Set the generic EA parameters
        self.n_children =self.n_pop
        
        # Get
        self.BNF_rules = aux.load_rules(rules_file)
        
        super().__init__(name) #Path(instance_file).stem
        self.parameters=parameters
        self.pop_decoded= [aux.decode(chromosome) 
                           for chromosome in self.pop.data]
    def init_pop(self):
        """
        Initialize the population sparse matrix with radnom integer vectors of 
        random lenght.
        """
        print("Initializing population...")
    
        self.pop = lil_matrix((self.n_pop, 15*self.max_kernels), dtype=np.int32)

        for i in range(self.n_pop):
            # Initialize
            n_kernels=random.randint(1,self.max_kernels//3)
            self.pop[i,:n_kernels*15]= np.random.randint(1, 256, 
                                                         size=(n_kernels*15))
   

    def f_fitfind_new(self, x):
        i, = np.where(self.pop == x)
        # The decoded function evaluated in the
        f_hat=self.pop_decoded[i]
        
        # # Objective 1: Absolute mean ponderated error of the pairs 
        # (f_hat(x_j),f(x_j))
        f_error=abs(f_hat.f(self.sample_points)-self.f_sample_point)      
        omega = self.K0 if f_error <= self.U else self.K1
        f_error=(omega @ f_error)/self.m
        # Objective 2: Penalty for the number of kernels
        n_kernels=len(f_hat.kernels)
        kernel_penalty=self.kernel_penalty_weight * n_kernels
        return f_error + kernel_penalty
    
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
        """
        Match the parents 2 by 2 randomly and return a 3d array to index
        """
        return parents.reshape(-1, 2)

    def crossover(self, parents):
        """
        Perform single-point crossover on the parents.
        """
        p1, p2 = parents
        # If crossover probability is not met or second parent is None, 
        # return parents
        if (random.random() >= self.pc or p2 is None):
            return p1, p2

        # Choose a random crossover point
        cross_point = np.random.randint(1, len(p1)//15)

        # Create two empty children
        c1 = np.zeros(len(p1), dtype=int)
        c2 = np.zeros(len(p2), dtype=int)

        # Perform the crossover
        c1[:cross_point], c1[cross_point:] = p1[:cross_point], p2[cross_point:]
        c2[:cross_point], c2[cross_point:] = p2[:cross_point], p1[cross_point:]

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
        idx = np.random.randint(0, n-1)
        # Swap the two selected genes
        x[idx] = random.randint(0,255)
        
        i, = np.where(self.pop == x)
        kernel_start=(i-i%15)
        kernel_position=kernel_start//15
        
        changed_kernel=aux.decode[x[kernel_start:kernel_start+15]]
        self.pop_decoded[i].kernels[kernel_position]=changed_kernel.kernels[0]
        self.pop_decoded[i].weights[kernel_position]=changed_kernel.weights[0]
        self.pop_adapts[i]=self.f_adapt(x)
        
        return x

    def select_survivors(self, children):
        """
        Substitute the previous popualtion with the children and apply elitism
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