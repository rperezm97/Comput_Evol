import numpy as np
import sys, os
import json
from scipy.stats.qmc import LatinHypercube
from multiprocessing import Pool, cpu_count
# We need to import modules from the current
# and parent directories.
A2_FOLDER = os.path.dirname(__file__)
sys.path.append(A2_FOLDER)
import aux_ES as aux
import funct
sys.path.append( os.path.join(A2_FOLDER, ".."))
from EA import Evolutive_algorithm

class Evolution_Strategy(Evolutive_algorithm):
    """
    Child class implementing an Evolution Strategy (GA) for optimizing a 
    multivariable function (expressed in a Function object, see funct.py).

    """
    def __init__(self, instance_id):
       """
        Initializes the class with the following attributes:
    
        - function: Function object to be optimized, as defined in funct.py
        -parameters_file: The path of a json file containing the parameters 
        as a dictionary {key:value}. If None, the default parameters will
        be loaded.
                  
       """
     
       print("Initializing the Evolutive Strategies algorithm for function optimization\n INSTANCE: {}".format(instance_id))
       # Function to be optimized
       function_name=instance_id.split("_")[0]
       self.function=getattr(funct, function_name)()
       
       
       super().__init__(instance_id, A2_FOLDER, 0,
                        convergence_thresholds=(0,0,1e-15))  # Path(instance_file).stem
       
    def init_parameters(self):
        #The id can be INS_PC_EXE and the parameter isnatce is just INS_PC, thus
        # we extract that
        parameter_instance="_".join(self.instance_id.split("_")[:2])
        parameters_file=os.path.join(A2_FOLDER,
                                "parameters/{}.json".format(parameter_instance))
        try:
            raise ValueError
            print("Loading parameters from file.")
            with open(parameters_file) as fp:
                parameters = json.load(fp)
        except:
            print("Invalid/empty parameter file.\nUsing default parameters.\n")
            parameters = {}
            parameters = {}
            
            parameters["n_gen"] = 1200
            parameters["n_dims"] = 10
            parameters["n_pop"] = 30
            parameters["n_children"] = 200
            parameters["crossover_type"] = "discrete"
            parameters["step_size"] = 10
            parameters["n_parents"] = 30
            parameters["tau"] = 0.0001
            parameters["tau_prime"] = 0.01
            parameters["selection_type"] = "+"
            parameters["eps_0"] = 0.00001
            print("Saving default parameters as {}.".format(parameters_file) ,
                   "Please modify the file adn re-run teh algorithm")
            
            #with open(parameters_file, "w") as fp:
            #    json.dump(parameters, fp)
        
        print("PARAMETERS={}\n".format(parameters))
        self.n_gen = parameters["n_gen"] 
        self.n_pop = parameters["n_pop"]
        self.n_dims = parameters["n_dims"]
        self.crossover_type=parameters["crossover_type"] 
        self.step_size = parameters["step_size"] 
        self.n_children = parameters["n_children"] 
        self.n_parents = parameters["n_parents"]
        self.tau=parameters["tau"] 
        self.tau_prime=parameters["tau_prime"] 
        self.selection_type=parameters["selection_type"]
        self.eps_0 = parameters["eps_0"]
    def init_pop(self):
        """
        Initialize the population.
        The decision variable vector x is initialized using Latin Hypercube Sampling
        (LHS) to ensure a well-distributed set of initial points within the defined domain.
        The strategy parameters s are initialized from a normal distribution with mean 1 and
        variance 1.
        """
        pop = np.empty((self.n_pop, 
                               self.n_dims+self.step_size))
        
        # Initialize the variable vector x. Since the lhs sreates a sample in a 
        # hypercube [0,1]^n, reescale according to the domain of the function
        sampler=LatinHypercube(d=self.n_dims)   
        diameter=(self.function.dom[1]-self.function.dom[0])                              
        pop[:,:self.n_dims]=(sampler.random(n=self.n_pop)-0.5)*diameter
        # Initialize the strategy parameters s 
        pop[:, self.n_dims:] = np.random.normal(1, 1, 
                                           (self.n_pop, self.step_size))
        return pop

    def f_fit(self, x):
        """Evaluate the objective function for a single individual or multiple 
        individuals.

        Arguments:
            x (np.ndarray): A single individual, represented by a (self.n_dims) 
            numpy array, or multiple individuals represented as an (N, self.n_dims)
            numpy array.

        Returns:
            float or np.ndarray: The evaluated objective function value(s).
        """
        # Take the variable vector(s) of the array of individuals/ 
        # the individual, and evaluate the function there
        x_val=x[:, :self.n_dims] if len(x.shape)==2 else x[:self.n_dims]
        
        return self.function.evaluate(x_val)-self.function.optimal
    
    def parent_selection(self):
        """
        Randomly select parents for the next generation.
        
        This function returns a sample of size lambda * rho (with replacement) from the
        current population, where lambda is the number of offspring to be generated in each
        generation and rho is the number of parents for each offspring.
        """
        # Do a random selection with replacement
        parent_indices = np.random.choice(self.n_pop,
                                      size=self.n_children * self.n_parents,
                                      replace=True)
        parent_matches= np.reshape(parent_indices, (self.n_children, self.n_parents))
        return parent_matches
    
    def variation_operators(self, parent_matches):
       # REECOMBINATION
        
        # Create an empty array for the offsprings
        children = np.empty((self.n_children, self.n_dims+self.step_size))
        matched_parents=self.pop[parent_matches]
        
        # First, do the crossover of the variables x of the parents
        x_matched_parents=matched_parents[:,:,:self.n_dims]
        children[:,:self.n_dims]=(aux.discrete_crossover(x_matched_parents)
                                    if self.crossover_type == "discrete" else 
                                   aux.inter_crossover(x_matched_parents))
      
        #Second, do the crossover of the strategy parameters
        
        s_matched_parents=matched_parents[:,:,self.n_dims:]
        children[:, self.n_dims:] = aux.inter_crossover(s_matched_parents)
        
        #MUTATION
        mutated_children= (aux.one_step_mutation(children,
                                                 self.tau,
                                                 self.eps_0,
                                                 self.function.dom) 
                          if self.step_size == 1 else 
                          aux.n_step_mutation(children,
                                                    self.tau,
                                                    self.tau_prime,
                                                    self.eps_0,
                                                    self.function.dom))
        # Clip the mutated
        
        return mutated_children
    
        
    def select_survivors(self, children):
        """
        Selects the survivors from the current population and the children.
    
        Args:
        children (ndarray): the generated children after crossover and mutation
        
        Returns:
        ndarray: the selected survivors to continue the evolution
        """
    
        # Calculate the fitness for the children
        children_fit = self.f_fit(children)
      
    
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
        sorted_indices = np.argsort(total_fit)
        self.pop = population[sorted_indices[:self.n_pop]]
        self.pop_fit = total_fit[sorted_indices[:self.n_pop]]
        
        self.best = 0
        self.best_adapt = self.pop_fit[self.best]
if __name__ == "__main__":

    a = Evolution_Strategy("fn4_P0")
    a.run()
