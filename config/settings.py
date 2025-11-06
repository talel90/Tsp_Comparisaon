"""Configuration par défaut pour les algorithmes TSP"""

# Paramètres du problème
DEFAULT_NUM_CITIES = 20
DEFAULT_SEED = 42

# Paramètres communs
DEFAULT_MAX_ITERATIONS = 500

# Paramètres Recherche Taboue
DEFAULT_TABU_SIZE = 30

# Paramètres Recuit Simulé
DEFAULT_INITIAL_TEMP = 1000
DEFAULT_COOLING_RATE = 0.995

# Paramètres Algorithme Génétique
DEFAULT_POPULATION_SIZE = 100
DEFAULT_MUTATION_RATE = 0.01

# Paramètres de visualisation
COLORS = {
    'Tabou Search': '#ff6b6b',
    'Recuit Simulé': '#4ecdc4',
    'AG (Roulette)': '#45b7d1',
    'AG (Rang)': '#96ceb4'
}

# Description des algorithmes
ALGORITHM_DESCRIPTIONS = {
    'Tabou Search': 'Recherche locale avec mémoire',
    'Recuit Simulé': 'Optimisation probabiliste inspirée de la métallurgie',
    'AG (Roulette)': 'Sélection proportionnelle à la fitness',
    'AG (Rang)': 'Sélection basée sur le classement'
}