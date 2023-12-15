import sys, os
import time
# We need to import modules from the current
# and parent directories.
A1_FOLDER = os.path.dirname(__file__)
sys.path.append(A1_FOLDER)
from GA import Genetic_Algorithm
sys.path.append( os.path.join(A1_FOLDER, ".."))
from test_EA import Test

def main():
    # print("Starting the application")
    # time.sleep(0.5)
    # instance = input("Select the problem instance (simple or complex): ")
    # time.sleep(0.5)
    # print(f"If necessary, modify the 3 parameter files param_{instance}.")
    # time.sleep(0.5)
    # print("Starting initial execution")
    # time.sleep(0.5)

    experiment = {}
    instance="Schwefel"
    for pm in ["B3", "G3",
               "G1", "B1", 
               "B2", "G2",
               
               ]:
        instance_id = f"{instance}_{pm}"
        #print(f"Initial execution case: {instance}, pc={int(pc)/10}")
        #name="GA_pc_{}".format(pc)
        model_GA=Genetic_Algorithm(instance_id)
        t = Test(model_GA, n_exe=6, initial_exe= 0)
        t.run_test()
    # print("Analyze the convergence plots in the 'plots' folder and decide "
    #       "if the number of generations is appropriate, too large, or too small")

    # ngen = 16000  # int(input("Enter the appropriate number of generations for evaluation:"))

    # results = []
    # for pc in [0.2, 0.5, 1]:
    #     param_file = f"./params/params_{instance}_pc{pc}.json"
    #     print(f"Evaluation experiment: {instance}, pc={pc}")
    #     results.append(experiment[pc].execute(ngen))
    #     time.sleep(0.5)

    # Test.plot_comparison(instance, experiment, ngen)


if __name__ == "__main__":
    main()
  
    
