## Activity 1: Genetic Algorithm for Traveling Salesman Problem
# Overview
This project aims to implement a genetic algorithm for solving the Traveling Salesman Problem (TSP). The TSP is a classic problem in computer science and operations research, where the goal is to find the shortest possible path that visits a set of cities exactly once, returning to the starting city. The problem is known to be NP-hard, which means that exact solutions are computationally infeasible for large problem sizes.

We will be implementing a genetic algorithm for solving the TSP, following the guidelines provided in the textbook "Fundamentos de la Computación Evolutiva" by Carmona and Galán. Specifically, we will be implementing an abstract class for a generic evolutive algorithm, and a Test class, that implements the evaluation loop for our EA class (and its children).

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
To evaluate the genetic algorithm, we will use the measures and types of graphs that appear in Chapter 9 of the textbook. For example, we will use the success rate and the progress curves to illustrate the performance of the genetic algorithm, along with other measures that we consider necessary. We will also measure the execution time of the algorithm.

We will evaluate the genetic algorithm using two different TSP instances: one with 100 cities and another with 10,000 cities. For each instance, we will vary the crossover probability and measure the success rate and the progress curves for each value of the crossover probability.

# Conclusion
In conclusion, this project aims to implement a genetic algorithm for solving the Traveling Salesman Problem, using the guidelines provided in the textbook "Fundamentos de la Computación Evolutiva" by Carmona and Galán. We will evaluate the performance of the genetic algorithm using two different TSP instances, and measure the success rate and
