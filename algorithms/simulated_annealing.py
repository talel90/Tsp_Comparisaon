import random
import numpy as np
from typing import List, Tuple
from algorithms.base_algorithm import BaseAlgorithm
from models.tsp_problem import TSPProblem

class SimulatedAnnealing(BaseAlgorithm):
    def __init__(self, problem: TSPProblem, max_iterations: int = 1000, 
                 initial_temp: float = 1000, cooling_rate: float = 0.995):
        super().__init__(problem)
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
    
    def get_neighbor(self, route: List[int]) -> List[int]:
        neighbor = route.copy()
        i, j = random.sample(range(len(route)), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor
    
    def solve(self) -> Tuple[List[int], float, List[float]]:
        current_solution = self.generate_initial_solution()
        best_solution = current_solution.copy()
        current_distance = self.problem.calculate_total_distance(current_solution)
        best_distance = current_distance
        
        temperature = self.initial_temp
        history = [best_distance]
        
        for _ in range(self.max_iterations):
            neighbor = self.get_neighbor(current_solution)
            neighbor_distance = self.problem.calculate_total_distance(neighbor)
            
            delta = neighbor_distance - current_distance
            
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                current_solution = neighbor
                current_distance = neighbor_distance
                
                if current_distance < best_distance:
                    best_solution = current_solution.copy()
                    best_distance = current_distance
            
            temperature *= self.cooling_rate
            history.append(best_distance)
            
            if temperature < 1e-10:
                break
        
        return best_solution, best_distance, history