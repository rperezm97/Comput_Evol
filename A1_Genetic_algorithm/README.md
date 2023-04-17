## Activity 1: Genetic Algorithm for Traveling Salesman Problem

# Overview
This project aims to implement a genetic algorithm for solving the Traveling Salesman Problem (TSP), which consists of finding the shortest route that visits a set of cities exactly once and returns to the starting city. The TSP is a classic problem in computer science and operations research, where the goal is to find the shortest possible path that visits a set of cities exactly once, returning to the starting city. 
The problem is known to be NP-hard, meaning that its solution time grows exponentially with the problem size. To tackle this challenge, we will use a Genetic Algorithm (GA), which is a type of Evolutionary Algorithm inspired by the process of natural selection. The GA will evolve a population of candidate solutions (i.e., tours) towards better solutions using genetic operators such as crossover and mutation.


# Files
The files for this activityconsist of:

- main.py: This file implements the genetic algorithm for TSP using the classes defined in the ea and test modules.
- ea.py: This file contains the implementation of the abstract EA class, which defines the basic structure of an evolutive algorithm, and the GA class, which implements the genetic algorithm for TSP.
- test.py: This file implements the Test class, which performs the evaluation loop of the genetic algorithm for TSP.
- params.json: This file contains the parameters used in the genetic algorithm, such as the population size, crossover and mutation probabilities, and the number of generations.


# Usage
To run the genetic algorithm for TSP, simply execute the main.py file with Python 3:
"""
python main.py
"""
The program will read the TSP instance from a file named instance.txt located in the same directory as main.py. The instance file should contain the coordinates of each city in a two-dimensional space. Each line of the file should be in the format x y, where x and y are the coordinates of a city. The first line should contain the number of cities in the instance.

The program will output the best solution found and its fitness value, as well as a plot of the fitness values of the best and average individuals over the generations.

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


# Evaluation
The test.py file performs the evaluation of the genetic algorithm using the parameters defined in the params.json file. The program evaluates the algorithm for two TSP instances: a simple one with 100 cities and a complex one with 10,000 cities.

The evaluation of the algorithm is performed using measures and types of graphs described in chapter 9 of the book by Carmona and Galán. The results are presented in the results folder, which contains the fitness values of the best and average individuals over the generations, as well as a plot of these values.

