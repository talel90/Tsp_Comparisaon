import random
from typing import List, Tuple
from algorithms.base_algorithm import BaseAlgorithm
from models.tsp_problem import TSPProblem

class GeneticAlgorithm(BaseAlgorithm):
    def __init__(self, problem: TSPProblem, population_size: int = 100, max_generations: int = 500,
                 mutation_rate: float = 0.01, selection_method: str = "roulette"):
        super().__init__(problem)
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
    
    def initialize_population(self) -> List[List[int]]:
        return [self.generate_initial_solution() for _ in range(self.population_size)]
    
    def calculate_fitness(self, population: List[List[int]]) -> List[float]:
        distances = [self.problem.calculate_total_distance(individual) for individual in population]
        max_dist = max(distances)
        return [max_dist - dist + 1 for dist in distances]
    
    def roulette_selection(self, population: List[List[int]], fitness: List[float]) -> List[List[int]]:
        total_fitness = sum(fitness)
        probabilities = [f / total_fitness for f in fitness]
        selected = []
        
        for _ in range(self.population_size):
            r = random.random()
            cumulative_prob = 0
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if r <= cumulative_prob:
                    selected.append(population[i].copy())
                    break
        
        return selected
    
    def rank_selection(self, population: List[List[int]], fitness: List[float]) -> List[List[int]]:
        sorted_population = [x for _, x in sorted(zip(fitness, population), reverse=True)]
        weights = [self.population_size - i for i in range(self.population_size)]
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        selected = []
        for _ in range(self.population_size):
            r = random.random()
            cumulative_prob = 0
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if r <= cumulative_prob:
                    selected.append(sorted_population[i].copy())
                    break
        
        return selected
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child = [None] * size
        child[start:end] = parent1[start:end]
        
        pointer = 0
        for i in range(size):
            if child[i] is None:
                while parent2[pointer] in child:
                    pointer += 1
                child[i] = parent2[pointer]
                pointer += 1
        
        return child
    
    def mutate(self, individual: List[int]) -> List[int]:
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual
    
    def solve(self) -> Tuple[List[int], float, List[float]]:
        population = self.initialize_population()
        best_individual = population[0].copy()
        best_distance = self.problem.calculate_total_distance(best_individual)
        
        history = [best_distance]
        
        for _ in range(self.max_generations):
            fitness = self.calculate_fitness(population)
            
            selected = (self.roulette_selection(population, fitness) 
                       if self.selection_method == "roulette" 
                       else self.rank_selection(population, fitness))
            
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % self.population_size]
                
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            
            current_best = min(population, 
                             key=lambda x: self.problem.calculate_total_distance(x))
            current_best_distance = self.problem.calculate_total_distance(current_best)
            
            if current_best_distance < best_distance:
                best_individual = current_best.copy()
                best_distance = current_best_distance
            
            history.append(best_distance)
        
        return best_individual, best_distance, history