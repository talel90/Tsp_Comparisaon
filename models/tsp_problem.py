import numpy as np
from typing import List, Tuple

class TSPProblem:
    def __init__(self, num_cities: int, seed: int = 42):
        self.num_cities = num_cities
        self.seed = seed
        self.cities = self.generate_cities()
        self.distance_matrix = self.calculate_distance_matrix()
    
    def generate_cities(self) -> List[Tuple[float, float]]:
        np.random.seed(self.seed)
        return [(np.random.uniform(0, 100), np.random.uniform(0, 100)) for _ in range(self.num_cities)]
    
    def calculate_distance_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    dx = self.cities[i][0] - self.cities[j][0]
                    dy = self.cities[i][1] - self.cities[j][1]
                    matrix[i][j] = np.sqrt(dx**2 + dy**2)
        return matrix
    
    def calculate_total_distance(self, route: List[int]) -> float:
        total = 0
        for i in range(len(route)):
            j = (i + 1) % len(route)
            total += self.distance_matrix[route[i]][route[j]]
        return total