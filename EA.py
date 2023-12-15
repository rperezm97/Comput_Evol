import numpy as np
from abc import ABC
import time
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm

class Evolutive_algorithm(ABC):
    """Abstract class that implements the methods of a generic evolutionary
    algorithm."""

    # ----------------INITIALIZATION OF VARIABLES AND PARAMETERS---------------
    def __init__(self, 
                 instance_id, 
                 base_folder=None,
                 convergence_threshold=1e-5,
                 sucess_threshold=1e-2):
        """Initialize the population and their fitness, the parameters and 
        other relevant variabels for the algorithm."""
        
        # Instance_id includes information about the insatcne and the run,
        # In the format instanceName_runId. It's used for loading parameters, 
        # logging, etc.
        self.instance_id = instance_id
        
        # TODO: is there a better way to execute?
        self.base_folder=base_folder
        
        # LOAD PARAMETERS:
        self.init_parameters()
       
        # INITIALIZE POPULATION VARIABLES:
        self.pop = self.init_pop()
        self.pop_fitness = self.f_fitness(self.pop)
        # Save the best individual with their adaptation value
        self.best = np.argmin(self.pop_fitness) 
        self.best_adapt = self.pop_fitness[self.best]
        
        # INITIALIZE THE CONVERGENCE/SUCESS THRESHOLD:
        self.convergence_threshold=convergence_threshold
        self.sucess_threshold=sucess_threshold
        
    def init_parameters(self):
        """Load parameters from the json file corresponding to the current 
        instance of the problem"""
        
         # Parameters are stored in the "parameters" folder, in a json file named 
         # after instanceName
        instanceName="_".join(self.instance_id.split("_")[:2])
        parameters_file=os.path.join(self.base_folder,
                                     f"parameters/{instanceName}.json")
        # Laod the parameters dictinary and hold it as an attribute of the class  
        with open(parameters_file) as f:
            parameters = json.load(f)
        # Use the keys in the parameter dict to define variables with the same 
        # name (for example, self.n_pop = self.parameters["n_pop"])    
        for key, value in parameters.items():
            setattr(self, key, value)
        print(parameters)
    
    # --------------------ABTRACT METHODS OF A GENERIC EA-----------------------
    def init_pop(self):
        """Initialize the population. To be implemented by child classes."""
        pass
    
    def f_fitness(self):
        """Calculate the fitness value of one or more individual. It
        should work with np.arrays or list of individuals.
        To be implemented by child classes."""
        pass

    def parent_selection(self):
        """Select parents from the population. It should return a list with
        the indices of the matched parents, ready for variation operators.
        To be implemented by child classes."""
        pass

    def variation_operators(self, parents):
        """Perform the variation operators (crossover and mutation, but could 
        be others) on two parents to create two children. To be
        implemented by child classes."""
        pass

    def select_survivors(self, children):
        """Select the individuals to be included in the next generation.
        To be implemented by child classes."""
        pass
    
    # -------------------------- MAIN EXECUTION CYCLE -------------------------- 
    # def convergence_condition(self,checkgen_i,bests, SDs):
    #     # We calculate the slope of teh curve, doin the discrete derivative
    #     # (implicitly, usinh h=check_step)
    #     gen=checkgen_i*self.n_gen//100
    #     change_rate=(bests[gen-1] - bests[gen])/(bests[gen-1])
    #     exploration_condition = change_rate<1e-5
        
    #     explotation_condition = SDs[checkgen_i] < 0.1*SDs[0]
        
    #     print("Rate of best_change:{}\n".format(
    #           change_rate))
    #     return exploration_condition and explotation_condition
        
    def check_success(self,checkgen_i, bests):
        
        if not (self.optimal is None):
            distance_optimal = np.abs(bests[checkgen_i] - self.optimal)-1
            condition_optimal = distance_optimal < self.optimal_threshold        

            print("Distance from bests towards know optimal: {}%".format(
                int(distance_optimal*100)))
            return condition_optimal 
        #return gen_convergence>0
    
    def run(self, early_stop=True):
        """
        Runs the evolutionary algorithm for ngen generations. In every iteration,
        it reproduces the population. Every ngen/100 iterations, it chekcs 
        convergence. If early_stop=True, it will stop when the convergence 
        condition has been archieved.
        
        Retuns the best and mean fitness values, with the standard deviation,
        of the 100 check generations (if early_stop=True, the values after 
        convergence will be the same as the one at the moment of convergence).
        """
        
        # INITIALIZE VARIABLES:
        
        # Output variables
        means = np.zeros(self.n_gen)
        bests = np.zeros(self.n_gen)
        SDs = np.zeros(100)
        # Log 
        log_file=os.path.join(self.base_folder,
                                f"logs/{self.instance_id}.txt")
        log = open(log_file,"w")
        # Convergence genration and time at the start of execution
        gen_conver=None
        t0 = time.time()

        # Check_step is the number of generations between convergence checkings.
        # It comes from dividing range(n_gen) in 100 equally spaced parts.
        check_step=(self.n_gen//100)
        
        print(f"Running {self.instance_id} for {self.n_gen} generations:\n")
        
        # MAIN EXECUTION CYCLE: 
        for gen in tqdm(range(self.n_gen)):
            # Select parents and apply the variation operators
            parent_matches = self.parent_selection()
            children=self.variation_operators(parent_matches)
            # Select survivors and update the population variables (in place)
            self.select_survivors(children)
            
            # Store best fitness value and the the empirical mean over 
            # the population  
            bests[gen] = self.best_adapt
            means[gen] = np.sum(self.pop_fitness) / self.n_pop
            
            # If gen is a multiple of the check_step, we check convergence
            if not (gen % check_step):
                #Get the index of the current checking generation (from 1 to 100)
                checkgen_i=gen//check_step
                
                # Calculate and store standard deviation (pseudovariance) of the
                # population fitness, aswell as the best value
                SDs[checkgen_i] = np.sqrt( np.sum ( ( self.pop_fitness 
                                                    - means[gen] )**2 )
                                         / ( self.n_pop - 1 ) )
                
                # Print the progress update and write the info to the log file
                print( (f"""  \n Gen {gen} :
                            \n Best = {bests[gen]} 
                            \n Mean = {means[gen]} 
                            \n STD = {SDs[checkgen_i]} 
                            \n \n """) ,
                      end="", flush=True)
                log.write(" ".join(map(str,[gen, 
                                            bests[gen], 
                                            means[gen], 
                                            SDs[checkgen_i], 
                                            "\n"])))

                # Check for convergence: in 100 generations, the best individual
                # hasn't improved (thus we have reached a plateau in exploration)
                # and the standard deviations hasn't changed (thus, the algorithm
                # has exploited that solution and t,e best individual has taken
                # over most of the population).
                # if (checkgen_i>0 and 
                #     self.convergence_condition(checkgen_i,
                #                                bests=bests, 
                #                                SDs=SDs)
                #     and not gen_conver):

                #     gen_conver = gen
                #     if early_stop:
                #         break
        
        # SAVING VALUES AND PLOTTING
        
        # Check the time of running
        t1 = time.time()
        t_conver = t1 - t0
        # Set the best and mean values of the generations after convergence as
        # the value of convergence, for the plots
        bests[gen:] = bests[gen]
        means[gen:] = means[gen]
        
        # Print the final results, formatting it accordingly to convergence 
        # being reached or not
        print(("\n Convergence"+
                   "was" if gen_conver else "wasn't"+ 
                f"""  reached in {t_conver} seconds, 
                    after {gen_conver or self.n_gen} 
                    generations.  
                    \n Fitness of the best individual: {self.best_adapt}"""), 
                flush=True)
        # Write the convergence info in the log
        log.write(" ".join(map(str,
                           [gen_conver, 
                            t_conver,
                            "\n"])))
        # Close log
        log.close()

        # Save the genome of the best individual, if save_best
        #TODO: log.write(str(self.pop[self.best]))
        
        # Plot the progress graph
        self.plot_convergence(gen_conver, bests, means, SDs)

        return self.pop[self.best], self.best_adapt, t_conver, gen_conver
    
    
    def plot_convergence(self, gen_converg, bests, means, SDs, visualize=False):
        """
        Plots the progress curve of the evolutive algorithm, including with the 
        value of  the best individual, the mean of the population, and the standard 
        deviation per generation.
        Also shows the point where the convergence condition is met (if it 
        doesn't converge, this point will be at zero).
        """
        
        
        n_gen_exe= gen_converg or self.n_gen
        generations = np.arange(n_gen_exe)
        # Create a list with the indices of the 100 reference generations
        ref_generations = [i for i in range(n_gen_exe) if not i % (self.n_gen//100)]
        # Create the figure
        plt.figure(figsize=(8, 6), dpi=200)
        plt.xlim(0,n_gen_exe)
        if self.optimal:
            minimum=self.optimal
        else:
            minimum=min(bests)
        plt.ylim(minimum-0.05*max(bests),1.05*max(bests))

        # Plot:
        # The mean
        
        plt.plot(generations,
                 means[:n_gen_exe], 
                 linewidth= .3,
                 color="orange",
                 label='Mean')
        # The standard deviation at the reference points
        plt.errorbar(ref_generations, 
                     means[ref_generations], 
                     yerr=SDs[:100*n_gen_exe//self.n_gen], 
                     color='Black',
                     elinewidth=.6,
                     capthick=.8, 
                     capsize=.8, 
                     alpha= .5,
                     fmt='o',
                     markersize=.7,
                     linewidth=.7)
        # The value of the best individual and the known optimal value
        plt.plot(generations,
                 bests[:n_gen_exe],
                 "--", 
                 linewidth= .8,
                 color="blue",
                 label='Best')
        
        if not (self.optimal is None):
            plt.plot(generations, 
                     self.optimal*np.ones(n_gen_exe),
                     label="Optimal", 
                     color="green")
        # The convergence generation on the best individual curve
        # plt.scatter(gen_converg,
        #            bests[gen_converg],
        #             color="red",
        #             label='Convergence', 
        #             zorder=1)

        # Add the grid, legend and labels
        
        plt.legend(loc='upper right', 
                   frameon=False,
                   fontsize=8)
        plt.xlabel('Generations')
        plt.ylabel('Adaptation value')

        plt.title('{} '.format(self.instance_id))
        
        #plt.yscale('log')
        # Create the axis
        plt.grid(True)
        
        # Save the figure in the plots folder and show it
        plt.savefig(os.path.join(self.base_folder,
                            'plots/{}.png'.format(self.instance_id)))
        
        # Show the plot
        if visualize:
            plt.show()