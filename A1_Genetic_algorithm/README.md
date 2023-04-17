## Activity 1: Genetic Algorithm for Traveling Salesman Problem

# Overview
This project aims to implement a genetic algorithm for solving the Traveling Salesman Problem (TSP), which consists of finding the shortest route that visits a set of cities exactly once and returns to the starting city. The TSP is a classic problem in computer science and operations research, where the goal is to find the shortest possible path that visits a set of cities exactly once, returning to the starting city. 
The problem is known to be NP-hard, meaning that its solution time grows exponentially with the problem size. To tackle this challenge, we will use a Genetic Algorithm (GA), which is a type of Evolutionary Algorithm inspired by the process of natural selection. The GA will evolve a population of candidate solutions (i.e., tours) towards better solutions using genetic operators such as crossover and mutation.


# Files
The files for this activity consist of:

- GA_TSP.py: This file implements the genetic algorithm for TSP using the classes defined in the EA and test modules in the parent folder.
- test.py: This file implements the Test class, which performs the evaluation loop of the genetic algorithm for TSP.
-aux.py: file with auxiliar functions for GA_TSP.
- main.py: This file runs statistical test for evaluating and comparing the genetic algorithm for both the simple and complex instance for three values of the parameter probability of crossover: pc=0.2,0.5 and 0.7.

The subfolder contained here are:

-instances: folder containing the coordinates of the cities for two instances of the problems, simple (100 cities) and comples (10000 cities) 
-params: folder with the json files containing a dict of parameters to be imported by TSP_GA.
-logs: folder containing the logs generated in the different runs of GA_TSP.
-plots: the plots of each run of a GA_TSP, or the statistical test for evaluation.

# Usage

Run main.py the statistical test for evaluating and comparing the genetic algorithm for both the simple and complex instance for three values of the parameter probability of crossover: pc=0.2,0.5 and 0.7. 

To run a the GA_TSP with a particular instance and a particualr set of parameters, you can call the method self.run() in an instance of a Genetic_Algorithm_TSP from GA_TSP.py. Read the docstring for more info about how to input the parameters and instances.

For doing a statatistical test for evaluation of a particualr set of parameters, instantiate a Genetic_Algorithm_TSP object and run the self.run_test() method of an instance of Test (selecting the instance of the Genetic_Algorithm_TSP and the number of executions for the test, ie, the size of the sample).

The evaluation of the algorithm is performed using measures and types of graphs described in chapter 9 of the book by Carmona and Galán. The results are presented in the results folders, which contains a log of the fitness values of the best and average individuals over the generations, as well as a plot of these values.

# Implementation
We will be implementing a genetic algorithm for solving the TSP, with the following specific characteristics:

- Representation of each individual: We will use a representation that associates a permutation of the cities to the genotype of an individual.
- Fitness function: The fitness function will assign to each individual the length of the path associated with its phenotype.
- Initialization of the population: We will initialize the population randomly, with the constraint that each city can only be visited once in each path.
- Parent selection: We will use tournament selection.
- Crossover: We will use partially mapped crossover with a given crossover probability.
- Mutation: We will use exchange mutation with a given mutation probability.
- Survivor selection: We will use a generational model, which replaces the entire current population with a new one. We will also apply elitism, so that the best individual in the current population is preserved in the next generation, provided that there is no individual in the new population with equal or better fitness value.
We will use two different TSP instances for testing our genetic algorithm: one with 100 cities and another with 10,000 cities. For each instance, we will vary the crossover probability and evaluate the performance of the genetic algorithm.


