import time
import os
import numpy as np
import matplotlib.pyplot as plt

class Test:
    def __init__(self, EA, n_exe, initial_exe, t_n):
        """
        Initialize the Test class.

        Args:
        EA (Evolutive_algorithm): The evolutive algorithm object.
        n_exe (int): Number of executions for the test.
        """
        
        
        print("\n\n -------- STARTING STATISTICAL TESTS ------")
        self.model_AE = EA
        self.instance_id=self.model_AE.instance_id
        self.n_exe = n_exe
        self.initial_exe=initial_exe
        
        self.MBF = np.empty(100)
        self.Mean_Error = np.empty(100)
        
        self.sample_bests=np.zeros((n_exe,100))  
        self.sample_means=np.zeros((n_exe,100))
        self.sample_DEs=np.zeros((n_exe,100))
        self.sample_gen_converg = np.zeros(n_exe, dtype=int)
        
        self.sample_best = np.zeros((n_exe, 100))
        self.sample_T_exe = np.empty(n_exe)
        self.t_n=t_n
        self.successes=np.empty(n_exe)
        self.SR = 0
        self.MT = 0

    def run_test(self):
        """
        Run a statistical test for evaluation of the evolutive algorithm (EA)
        with a sample of n_exe executions.
        """
        print("\nStarting parallel executions...")
        t1 = time.time()
        for i in range(self.initial_exe,self.n_exe):
              np.random.seed(i)
             
              self.model_AE.instance_id = self.instance_id+"_EXE{}".format(i)
              self.model_AE.pop = self.model_AE.init_pop()
              self.model_AE.pop_fit = self.model_AE.f_fit(self.model_AE.pop)
              self.model_AE.best_adapt = np.min(self.model_AE.pop_fit)
              self.model_AE.best = np.argmin(self.model_AE.pop_fit)
              print("----------------EXECUTION {}---------------".format(i))
              self.model_AE.run()
            
        t2 = time.time()
        t_exe = t2 - t1
        
        print("Executions {} to {} finished, time eluted= {}".format(self.initial_exe,
                                                              self.n_exe-1,
                                                              t_exe))
        self.estimate_indices()
        
    def estimate_indices(self):    
        log_folder=os.path.join(self.model_AE.base_folder,"logs") 
        execution_logs=[log_file for log_file in os.listdir(log_folder)
                         if ((log_file[:len(self.instance_id)]==self.instance_id) 
                         and int(log_file[-5])<self.n_exe)]
        if len(execution_logs)<self.n_exe:
            print("Some execution is missing, check the folder")
            return
        
        for exe_i,log_file in enumerate(execution_logs):
        
            with open(os.path.join(log_folder,log_file)) as fp:
                f = fp.readlines()
                for j,line in enumerate(f):
                    if line[0]=="[":
                        break
                
                (_,
                 self.sample_bests[exe_i,:],
                 self.sample_bests[exe_i,:],
                 self.sample_bests[exe_i,:]
                )= np.loadtxt(f[:j-1], dtype=np.float32)
                gen_converg, t_converg,_= f[j-1].split(" ")
                
                gen_converg=int(gen_converg)
                t_conver=float(t_converg)
                
            self.sample_gen_converg[exe_i]=gen_converg
            
            n_gen_exe= gen_converg if gen_converg>0 else self.model_AE.n_gen
            t=100*n_gen_exe//self.model_AE.n_gen
            self.sample_best[exe_i,:t+1]=self.sample_bests[exe_i,:]
            self.sample_best[exe_i,t+1:]=self.sample_bests[exe_i,:][t]
            self.sample_T_exe[exe_i]=t_conver
            self.successes[exe_i]=self.model_AE.check_success(self.sample_bests[exe_i,:],
                                                    gen_converg,
                                                    t_converg)
        
        self.MBF = np.sum(self.sample_best, axis=0) / self.n_exe
        print("\nMBF: {}%".format(self.MBF[-1]))
        ngen = self.MBF.shape[0]
        idx = [i for i in range(ngen) if not i % (ngen // 100)]
        
        #Confidence_interval
        s_n = np.sqrt(np.sum((self.sample_best[:, idx] -
                              self.MBF[idx]
                              ) ** 2
                             , axis=0
                             ) / (self.n_exe - 1))
        # ME at 95%
        self.Mean_Error = self.t_n * s_n / np.sqrt(self.n_exe-1)
        self.SR= np.sum(self.successes)/ self.n_exe
        print("\nSuccess rate: {}%".format(100 * self.SR))
        
        self.MT = np.sum(self.sample_T_exe*self.successes) 
        if self.MT:
            self.MT/=(np.sum(self.successes))
            
        print("\nAverage time: {}s".format(self.MT))


    def plot_MBF(self, ax=None):
        """
        Add the MBF plot to the given axis.

        Args:
        ax (matplotlib.axis): The axis to plot on.
        experiment (Test): The Test object containing the experimental results.
        label (str): The label for the plot.
        """
        
        ngen = len(self.MBF)
        
        adding_plot= ax
        if not adding_plot:
            ax = plt.figure(figsize=(8, 6), dpi=200)
            plt.grid(True)
            plt.xlim(0, ngen)
            plt.xlabel('Generations')
            plt.ylabel('Fitness Value')
            plt.title('Statistical estimation for {} executions)'.format(self.instance_id, self.n_exe))
            plt.grid(True)
        
        x = np.arange(ngen)

        plt.plot(x, self.MBF,
                linewidth=1,
                label="MBF of {} (SR={}, MT={}s)".format(self.instance_id, self.SR, int(self.MT)))

        plt.errorbar(x, self.MBF,
                    yerr=self.Mean_Error, color='Black',
                    elinewidth=.6, capthick=.8,
                    capsize=.8, alpha=0.5,
                    fmt='o', markersize=.7, linewidth=.7)
        if adding_plot:
            return ax
        else:
            plt.legend()
            plt.show()
        
if __name__=="__main__":
    import sys
    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    print(current)
    sys.path.append(current)
    sys.path.append(parent)
    from A1_Genetic_algorithm.GA_TSP import Genetic_Algorithm_TSP
    ga=Genetic_Algorithm_TSP("complex_02")
    t=Test(ga, n_exe=5, initial_exe=0,t_n=2)
    t.run_test()
    #t.plot_MBF()