from typing import List, Tuple
from algorithms.base_algorithm import BaseAlgorithm
from models.tsp_problem import TSPProblem

class TabuSearch(BaseAlgorithm):
    def __init__(self, problem: TSPProblem, max_iterations: int = 1000, tabu_size: int = 50):
        super().__init__(problem)
        self.max_iterations = max_iterations
        self.tabu_size = tabu_size
        self.tabu_list = []
    
    def get_neighbors(self, route: List[int]) -> List[List[int]]:
        neighbors = []
        for i in range(len(route)):
            for j in range(i + 1, len(route)):
                neighbor = route.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)
        return neighbors
    
    def solve(self) -> Tuple[List[int], float, List[float]]:
        current_solution = self.generate_initial_solution()
        best_solution = current_solution.copy()
        best_distance = self.problem.calculate_total_distance(best_solution)
        
        history = [best_distance]
        
        for _ in range(self.max_iterations):
            neighbors = self.get_neighbors(current_solution)
            best_neighbor = None
            best_neighbor_distance = float('inf')
            
            for neighbor in neighbors:
                neighbor_distance = self.problem.calculate_total_distance(neighbor)
                
                move = tuple(sorted([neighbor.index(current_solution[i]) 
                                   for i in range(len(current_solution)) 
                                   if neighbor[i] != current_solution[i]]))
                
                if move in self.tabu_list and neighbor_distance >= best_distance:
                    continue
                
                if neighbor_distance < best_neighbor_distance:
                    best_neighbor = neighbor
                    best_neighbor_distance = neighbor_distance
            
            if best_neighbor is None:
                break
                
            move = tuple(sorted([best_neighbor.index(current_solution[i]) 
                               for i in range(len(current_solution)) 
                               if best_neighbor[i] != current_solution[i]]))
            self.tabu_list.append(move)
            if len(self.tabu_list) > self.tabu_size:
                self.tabu_list.pop(0)
            
            current_solution = best_neighbor
            current_distance = best_neighbor_distance
            
            if current_distance < best_distance:
                best_solution = current_solution.copy()
                best_distance = current_distance
            
            history.append(best_distance)
        
        return best_solution, best_distance, history