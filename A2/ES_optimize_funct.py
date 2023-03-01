import numpy as np
from ypstruct import structure
import random
from multiprocessing import Pool
from functools import partial
from pop import AE
from scipy.stats.qmc import LatinHypercube
from pop import Evolutive_algorithm

class Evolution_Strategy(Evolutive_algorithm):
    """Implementation of the Evolution Strategy algorithm.
    
    This class inherits from the Evolutive_algorithm abstract class and 
    implements the required methods to run the Evolution Strategy algorithm.

    Attributes:
        function (Function): The function to be optimized.
        n_pop (int): Number of individuals in the population.
        n: Size of the variable vector x
        mutation_step (int): Number of sigma values of the individual (strategy 
                     parameters s).
        d (int): Total dimension of an individual.
        num_children (int): Number of offsprings created in each generation.
        num_parents (int): Number of parents involved in the creation of an 
                           offspring.
        tau (float): Hyperparameter for individual mutation of s
        tau_prime (float): Hyperparameter for global change in mutability.
        eps_0 (float): Lower threshold for strategy parameters.
        selection_type (str): the type of survivor selection to use
    """
    def __init__(self, function, params):
       """
        Initializes the class with the following attributes:
    
        function: Function object to be optimized, defined in funct.py
        params: Dictionary of parameters, with the following keys:
            nu: Number of individuals in the population
            n: Dimension of the variables of the function to optimize
            mutation_step: Number of sigma values of the individual (strategy 
                   parameters). It can be "1" for non-correlated 1-step size 
                   mutation, or "n" for non-correlated n-step size mutation
            lambda: Number of offsprings generated
            rho: Number of parents involved in the creation of an offspring. It
                 should be greater than 2, since we're performing global 
                 crossover
            crossover_type: String that determines the type of crossover 
                operator. Can be "discrete" or "intermediate"
            tau(float): Hyperparameter that determines the step for the 
                mutation of the strategy parameters s.It is the standard 
                deviation of the log-normal distribution that generates a 
                perturbation of each of the strategy parameters independently.
            tau_prime(float): Hyperparameter that determines the step for 
                the mutation of the strategy parameters. It is the standard 
                deviation of the log-normal distribution that generates the 
                global change in the mutability. Only needed for the 
                non-correlated n-step size mutation, to preserve the degrees 
                of freedom.
            eps_0(float): Hyperparameter that determines the lower threshold
                          for the strategy parameters, so they don't reach 0.
            selection_type (str): the type of survivor selection to use, either 
                         "+" or ","
                  
       """
       # Function to be optimized
       self.function = function
       # Number of individuals in the population
       self.n_pop = params["nu"]
       #Number of dimensions of the variables of the function to optimize
       self.n = params["n"]
       # Number of sigma values of the individual (strategy parameters)
       self.mutation_step = self.n if params["mutation_step"] else 1
       # Total dimension of an individual
       self.d = self.n + self.mutation_step
       # Number of offsprings generated
       self.num_children = params["lambda"]
       # Number of parents involved in the creation of an offspring
       self.num_parents = params["rho"]
       #The type of crossover (intermediate or discrete)
       self.crossover_type= params["crossover_type"]
       # Hyperparameter for individual mutation of strategy parameters
       self.tau = params["tau"]
       # Hyperparameter for global change in mutability
       self.tau_prime = 0 if self.mutation_step == 1 else params["tau_prime"]
       # Lower threshold for strategy parameters to prevent 0
       self.eps_0 = params["eps_0"]
       #Survivor selection type
       self.selection_type=params["selection_type"]
    def init_pop(self):
        """
        Initialize the population.The variable vector x is initialized using 
        Latin Hypercube Sampling (LHS) method to ensure a uniform distribution 
        within the defined domain. The strategy parameters s are initialized 
        randomly, uniformly between epsilon_0 and 10.
        """
        population = np.empty((self.n_pop, self.d))
        
        # Initialize the variable vector x. Since the lhs sreates a sample in a 
        # hypercube [0,1]^n, reescale according to the domain of the function
        sampler=LatinHypercube(d=self.n)   
        diameter=(self.function.dom[1]-self.function.dom[0])                                 
        mean=self.function.dom[0]+diameter//2                                 
        population[:,:self.n]=(sampler.random(n=self.n_pop))*diameter-mean
        # Initialize the strategy parameters s 
        population[:, self.n:] = np.random.uniform(self.eps_0, 10, 
                                                   (self.n_pop, self.mutation_step))
        self.population = population


    def f_adapt(self, ind):
        """Evaluate the objective function for a single individual or multiple 
        individuals.

        Arguments:
            x (np.ndarray): A single individual, represented by a (self.d) 
            numpy array, or multiple individuals represented as an (N, self.d)
            numpy array.

        Returns:
            float or np.ndarray: The evaluated objective function value(s).
        """
        # Take the variable vector(s) of the array of individuals/ 
        # the individual, and evaluate the function there
        x=ind[:, :self.n] if len(ind.shape)==2 else ind[:self.n]
        
        return self.function.f(x)
    
    def select(self):
         """
         Randomly sample the parents for each generation from the pop.
         
         It returns a sample of lambda*rho individuals (with repetition), since 
         each rho parents generate 1 offpring, and the algorithm must create
         lambda offsprings in each generation
         """
         # Do a random selection with replacement
         return np.random.choice(self.n_pop, 
                          size=self.num_parents*self.num_children , 
                          replace=True)

    def crossover(self, parents):
        """
        Perform crossover on a set of parent individuals to produce offspring.
        For the variables of the parents, it can perform discrete crossover or 
        intermediate promediate crossover, depending on the value indicated 
        in params["crossover_type"]. For the strategy parameters, it always 
        does intermediate promediate recombination.
        
        Parameters
        ----------
        parents : numpy.ndarray
            The parent individuals.
            
        Returns
        -------
        numpy.ndarray
            The offspring generated from the parent individuals.
        """
        # Create an empty array for the offsprings
        offsprings = np.empty((self.num_children, self.d))
        # First, do the crossover of the variables x of the parents
        x_parents=parents[:,:,:self.n]
        if self.crossover_type == "discrete":
            # Create an array of the form [[r(i,j) for 0<j<n] for 0<i<lambda]
            # where r(i,j) is a random number between 0 and num_parents-1, 
            # that represents the index of the parent that will provide the 
            # j-th variable to the i-th offspring. 
            variable_idx=np.random.randint(low=0,high=self.num_parents, 
                                               size=(self.num_children,self.n))
            # Use advanced numpy indexing to generate x_children[i, j] = 
            # x_parents[i, variable_idx[i, j], j]
            i=np.arange(self.num_children).reshape(self.num_children, 1)
            j=np.arange(self.n)
            offsprings[:, :self.n] = x_parents[i,variable_idx,j]
        elif self.crossover_type == "intermediate":
            # Do the mean of the parents for the generation of each child
            offsprings[:, :self.n] = np.mean(x_parents, axis=1)
        else:
            raise ValueError("Invalid crossover type: {}".format(
                                                          self.crossover_type))
        
        #Second, do the crossover of the strategy parameters
        offsprings[:, self.n:] = np.mean(parents[:,:,self.n:], axis=1)
        return offsprings
    
    def mutate(self, children):
        """
        Mutates the population with 1-step or n-step mutation.
         
        Parameters:
        -----------
        children : np.ndarray, shape (num_children, n + mutation_step)
            Array of children generated by crossover.
         
        Returns:
        --------
        np.ndarray, shape (num_children, n + mutation_step)
            Mutated children array.
        """
        n = self.n
        tau = self.tau
        tau_prime = self.tau_prime
        mut=children.copy()
        n_children = mut.shape[0]
        
    
        # Mutatethe population with 1-step or n-step mutation
        if self.mutation_step == "1":
            # For the variable mutations, generate a matrix of perturbations 
            # from a normal N(0,1) and multiply each row by the corresponding 
            # sigma value
            mut[:,:n]+= mut[:, :, n] * np.random.normal(size=(n_children, n))
            # For the strategy parameter mutations, generate a vector of 
            # perturbations from a normal N(0,1), and multiply 
            # it by tau. Then exponentiate this to get the log normal.
            mut[:,n] *= np.exp(tau * np.random.normal(size=(n_children,)))
        elif self.mutation_step == self.n:
            # For the variable mutations, generate a list of perturbation 
            # vectors, where each vector comes from a multivariate N(0,C), 
            # where C is the diagonal covariant matrix defined by the strategy
            # parameters
            mut[:,:n]+= [np.random.multivariate_normal(mean=np.zeros(self.n),  
                                                      cov=np.diag(mut[i,n:]), 
                                                      size=self.num_children) 
                         for i in range(self.n_pop)]
            
            # For the strategy parameter mutations, generate the global and 
            # individual normal perturbations with tau and tau', sum them
            # with numpy broadcast and exponentiate them to get the total 
            # perturbation
            global_mut=tau_prime * np.random.normal(size=(self.num_children,1))
            local_mut=tau * np.random.normal(size=(self.num_children, n))
            mut[:,n:] *= np.exp(global_mut+local_mut)
    
        # Combine variable vector and strategy parameters
        
        mut[:,:n] = np.clip(mut[:,:n], 
                            a_min=self.function.dom[0], 
                            a_max=self.function.dom[1])
        mut[:,n:]= np.clip(mut[:,n:], 
                            a_min=self.eps_0)
    
        return mut
    
   
    def select_survivors(self, children):
        """
        Selects the survivors from the current population and the children.
    
        Args:
        children (ndarray): the generated children after crossover and mutation
        
        Returns:
        ndarray: the selected survivors to continue the evolution
        """
    
        # Calculate the fitness for the children
        children_fit = self.function.f(children)
      
        # Calculate the total fitness of the combined population
        total_fit = np.concatenate((self.pop_fit, children_fit))
    
    
        # Select the survivors based on the selection type
        if self.selection_type == ",":
            population = children
            total_fit = children_fit

        elif self.selection_type == "+":
            # Concatenate the current population and children
            population = np.concatenate((self.pop, children))
            total_fit = np.concatenate((self.pop_fit, children_fit))
            
        else:
            raise ValueError("Invalid selection_type: must be '+' or ','")
            
        # Sort the population by fitness in descending order
        sorted_indices = np.argsort(total_fit)[::-1]
        survivors = population[sorted_indices[:self.n_pop]]
        
        return survivors