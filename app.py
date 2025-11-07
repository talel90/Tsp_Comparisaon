import streamlit as st
import matplotlib
# Use a non-interactive backend for headless deployments (prevents errors when no GUI is available)
try:
    matplotlib.use('Agg')
except Exception:
    # If backend can't be set, continue; pyplot import may still work depending on environment
    pass
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

from models.tsp_problem import TSPProblem
from algorithms.tabu_search import TabuSearch
from algorithms.simulated_annealing import SimulatedAnnealing
from algorithms.genetic_algorithm import GeneticAlgorithm
from utils.visualization import plot_solution, plot_convergence, plot_performance_comparison
from config.settings import *

def load_custom_css():
    with open("static/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Comparaison Algorithmes TSP",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    load_custom_css()
    
    st.markdown(
        '<h1 class="main-title">🎯 Comparaison des Algorithmes TSP</h1>',
        unsafe_allow_html=True
    )
    
    st.markdown(
        '<div class="intro-text">Cette application compare 4 algorithmes d\'optimisation '
        'pour résoudre le problème du voyageur de commerce :</div>',
        unsafe_allow_html=True
    )
    
    for algo, desc in ALGORITHM_DESCRIPTIONS.items():
        st.markdown(f"- **{algo}** : {desc}")
    
    # Sidebar pour les paramètres avec style personnalisé
    st.sidebar.markdown(
        '<h2 style="color: #4f46e5; border-bottom: 2px solid #4f46e5; padding-bottom: 0.5rem;">'
        'Paramètres du problème</h2>',
        unsafe_allow_html=True
    )
    
    with st.sidebar:
        num_cities = st.slider(
            "Nombre de villes",
            10, 50, DEFAULT_NUM_CITIES,
            help="Ajustez le nombre de villes dans le problème"
        )
        
        seed = st.number_input(
            "Seed aléatoire",
            value=DEFAULT_SEED,
            help="Définissez une valeur pour reproduire les mêmes résultats"
        )
    
    st.sidebar.header("Paramètres des algorithmes")
    
    # Paramètres communs
    max_iterations = st.sidebar.slider("Nombre maximum d'itérations", 
                                     100, 2000, DEFAULT_MAX_ITERATIONS)
    
    # Paramètres spécifiques
    tabu_size = st.sidebar.slider("Taille liste taboue", 
                                 10, 100, DEFAULT_TABU_SIZE)
    initial_temp = st.sidebar.slider("Température initiale", 
                                   100, 5000, DEFAULT_INITIAL_TEMP)
    cooling_rate = st.sidebar.slider("Taux de refroidissement", 
                                   0.990, 0.999, DEFAULT_COOLING_RATE, 0.001)
    population_size = st.sidebar.slider("Taille population (AG)", 
                                      50, 200, DEFAULT_POPULATION_SIZE)
    mutation_rate = st.sidebar.slider("Taux de mutation (AG)", 
                                    0.001, 0.1, DEFAULT_MUTATION_RATE, 0.001)
    
    if st.sidebar.button("Lancer la comparaison"):
        with st.spinner("Calcul en cours..."):
            # Initialiser le problème
            problem = TSPProblem(num_cities, seed)
            
            # Initialiser les algorithmes
            algorithms = {
                "Tabou Search": TabuSearch(problem, max_iterations, tabu_size),
                "Recuit Simulé": SimulatedAnnealing(problem, max_iterations, 
                                                  initial_temp, cooling_rate),
                "AG (Roulette)": GeneticAlgorithm(problem, population_size, 
                                                 max_iterations//2, mutation_rate, "roulette"),
                "AG (Rang)": GeneticAlgorithm(problem, population_size, 
                                            max_iterations//2, mutation_rate, "rank")
            }
            
            results = {}
            execution_times = {}
            
            # Exécuter les algorithmes
            for name, algorithm in algorithms.items():
                start_time = time.time()
                solution, distance, history = algorithm.solve()
                end_time = time.time()
                
                results[name] = {
                    'solution': solution,
                    'distance': distance,
                    'history': history,
                    'time': end_time - start_time
                }
                execution_times[name] = end_time - start_time
            
            # Afficher les résultats
            st.markdown('<h2 class="section-header">📊 Résultats de la comparaison</h2>', 
                       unsafe_allow_html=True)
            
            # Métriques comparatives avec style personnalisé
            metric_container = st.container()
            with metric_container:
                col1, col2, col3, col4 = st.columns(4)
                
                best_algorithm = min(results.keys(), 
                                   key=lambda x: results[x]['distance'])
                best_distance = results[best_algorithm]['distance']
                
                metric_style = """
                <div class="metric-card">
                    <h3 style="color: #4f46e5; margin: 0;">{value}</h3>
                    <p style="color: #6b7280; margin: 0.5rem 0 0 0;">{label}</p>
                </div>
                """
            
            with col1:
                st.markdown(
                    metric_style.format(
                        value=best_algorithm,
                        label="Meilleur algorithme"
                    ),
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    metric_style.format(
                        value=f"{best_distance:.2f}",
                        label="Meilleure distance"
                    ),
                    unsafe_allow_html=True
                )
            with col3:
                fastest_algorithm = min(execution_times.keys(), 
                                     key=lambda x: execution_times[x])
                st.markdown(
                    metric_style.format(
                        value=fastest_algorithm,
                        label="Plus rapide"
                    ),
                    unsafe_allow_html=True
                )
            with col4:
                st.markdown(
                    metric_style.format(
                        value=f"{np.mean(list(execution_times.values())):.2f}s",
                        label="Temps moyen"
                    ),
                    unsafe_allow_html=True
                )
            
            # Tableau comparatif détaillé
            st.markdown('<h3 class="section-header">Tableau comparatif détaillé</h3>',
                       unsafe_allow_html=True)
            comparison_data = []
            for name, result in results.items():
                comparison_data.append({
                    'Algorithme': name,
                    'Distance': f"{result['distance']:.2f}",
                    'Temps (s)': f"{result['time']:.2f}",
                    'Itérations': len(result['history']),
                    '% du meilleur': f"{(result['distance']/best_distance - 1)*100:.1f}%"
                })
            
            st.table(pd.DataFrame(comparison_data))
            
            # Graphiques
            st.markdown('<h3 class="section-header">Visualisation des solutions</h3>',
                       unsafe_allow_html=True)
            
            with st.container():
                # Configure plotting style: prefer seaborn if available, otherwise fall back to a
                # built-in matplotlib style to avoid crashing when a style is unavailable.
                try:
                    import seaborn as sns
                    sns.set_style("whitegrid")
                    sns.set_palette("husl")
                except Exception:
                    # If seaborn isn't available or fails for any reason, use a safe matplotlib style
                    try:
                        plt.style.use('ggplot')
                    except Exception:
                        # As a final fallback use the default style
                        plt.style.use('default')

                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                axes = axes.flatten()

                # Configuration supplémentaire pour améliorer l'apparence
                for ax in axes:
                    ax.grid(True, alpha=0.3)
                    ax.set_facecolor('#f8f9fa')
            
            for idx, (name, result) in enumerate(results.items()):
                plot_solution(problem, result['solution'], 
                            f"{name}\nDistance: {result['distance']:.2f}", 
                            axes[idx])
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Courbes de convergence
            st.subheader("Courbes de convergence")
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_convergence(results, ax)
            st.pyplot(fig)
            
            # Analyse statistique
            st.markdown('<h3 class="section-header">Analyse statistique</h3>',
                       unsafe_allow_html=True)
            
            stats_container = st.container()
            with stats_container:
                col1, col2 = st.columns(2)
            
            with col1:
                # Temps d'exécution
                fig, ax = plt.subplots(figsize=(8, 4))
                plot_performance_comparison(execution_times, "Temps (secondes)", ax)
                st.pyplot(fig)
            
            with col2:
                # Distances finales
                fig, ax = plt.subplots(figsize=(8, 4))
                distances = {name: result['distance'] for name, result in results.items()}
                plot_performance_comparison(distances, "Distance", ax)
                st.pyplot(fig)

def footer():
    st.markdown("""
    <div style="position: fixed; bottom: 0; width: 100%; background-color: #f8f9fa; 
                padding: 1rem; text-align: center; border-top: 1px solid #e5e7eb;">
        <p style="color: #6b7280; margin: 0;">
            Développé avec ❤️ | 
            <a href="https://github.com" target="_blank" style="color: #4f46e5; text-decoration: none;">
                Code source
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    footer()