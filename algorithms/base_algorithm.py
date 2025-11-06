from abc import ABC, abstractmethod
from typing import List, Tuple
import random
from models.tsp_problem import TSPProblem

class BaseAlgorithm(ABC):
    def __init__(self, problem: TSPProblem):
        self.problem = problem

    def generate_initial_solution(self) -> List[int]:
        route = list(range(self.problem.num_cities))
        random.shuffle(route)
        return route

    @abstractmethod
    def solve(self) -> Tuple[List[int], float, List[float]]:
        """
        Résout le problème TSP
        Returns:
            Tuple[List[int], float, List[float]]: (meilleure solution, meilleure distance, historique)
        """
        pass