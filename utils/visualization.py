from typing import List
import matplotlib.pyplot as plt
from models.tsp_problem import TSPProblem

def plot_solution(problem: TSPProblem, solution: List[int], title: str, ax):
    """Affiche la solution TSP"""
    cities = problem.cities
    route = [cities[i] for i in solution] + [cities[solution[0]]]
    
    x, y = zip(*route)
    ax.plot(x, y, 'o-', markersize=8, linewidth=2, alpha=0.7)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Ajouter les numéros des villes
    for i, (city_x, city_y) in enumerate(cities):
        ax.annotate(str(i), (city_x, city_y), xytext=(5, 5), textcoords='offset points')

def plot_convergence(results: dict, ax):
    """Trace les courbes de convergence"""
    for name, result in results.items():
        ax.plot(result['history'], label=name, alpha=0.8)
    
    ax.set_xlabel('Itérations')
    ax.set_ylabel('Distance')
    ax.set_title('Convergence des algorithmes')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_performance_comparison(data: dict, metric: str, ax):
    """Trace un graphique en barres pour comparer les performances"""
    names = list(data.keys())
    values = list(data.values())
    bars = ax.bar(names, values, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} par algorithme')
    plt.xticks(rotation=45)