import sys, os
import time
# We need to import modules from the current
# and parent directories.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
print(current)
sys.path.append(current)
sys.path.append(parent)

from test_EA import Test
from GA_TSP import Genetic_Algorithm_TSP

def main():
    print("Starting the application")
    time.sleep(0.5)
    instance = input("Select the problem instance (simple or complex): ")
    time.sleep(0.5)
    print(f"If necessary, modify the 3 parameter files param_{instance}.")
    time.sleep(0.5)
    print("Starting initial execution")
    time.sleep(0.5)

    experiment = {}
    for pc in [0.2, 0.5, 1]:
        parameters_file = f"./params/params_{instance}_pc{pc}.json"
        print(f"Initial execution case: {instance}, pc={pc}")
        name="GA_pc_{}".format(pc)
        Genetic_Algorithm_TSP(name, instance, parameters_file)
        experiment[pc] = Test(EA, n_exe=1)
        gen_converg = experiment[pc].estimate_Tconverg()
        print(f"Converges at generation {gen_converg}")
        time.sleep(0.5)

    print("Analyze the convergence plots in the 'plots' folder and decide "
          "if the number of generations is appropriate, too large, or too small")

    ngen = 16000  # int(input("Enter the appropriate number of generations for evaluation:"))

    results = []
    for pc in [0.2, 0.5, 1]:
        param_file = f"./params/params_{instance}_pc{pc}.json"
        print(f"Evaluation experiment: {instance}, pc={pc}")
        results.append(experiment[pc].execute(ngen))
        time.sleep(0.5)

    Test.plot_comparison(instance, experiment, ngen)


if __name__ == "__main__":
    main()
  
    
