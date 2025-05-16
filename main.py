import numpy as np
import time
from typing import Dict, List, Any
import pandas as pd
from tqdm import tqdm

from utils.problem import KnapsackProblem, generate_random_problem
from utils.visualization import (plot_convergence, plot_boxplot, 
                               create_comparison_table, plot_solution_distribution)

# Importar os algoritmos (serão implementados depois)
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.ant_colony import AntColony
from algorithms.particle_swarm import ParticleSwarm
from algorithms.cuckoo_search import CuckooSearch

def run_experiment(problem: KnapsackProblem,
                  n_runs: int = 30,
                  max_iterations: int = 100) -> Dict[str, Dict[str, Any]]:
    """
    Executa experimentos comparando diferentes algoritmos.
    
    Args:
        problem: Instância do problema da mochila
        n_runs: Número de execuções para cada algoritmo
        max_iterations: Número máximo de iterações por execução
        
    Returns:
        Dicionário com resultados detalhados de cada algoritmo
    """
    algorithms = {
        'GA': GeneticAlgorithm(problem),
        'ACO': AntColony(problem),
        'PSO': ParticleSwarm(problem),
        'CS': CuckooSearch(problem)
    }
    
    results = {}
    
    for algo_name, algorithm in algorithms.items():
        print(f"\nExecutando {algo_name}...")
        
        algo_results = {
            'values': [],
            'times': [],
            'solutions': [],
            'history': []
        }
        
        for _ in tqdm(range(n_runs)):
            start_time = time.time()
            
            # Executa o algoritmo
            best_solution, best_value, history = algorithm.run(max_iterations)
            
            end_time = time.time()
            
            # Armazena os resultados
            algo_results['values'].append(best_value)
            algo_results['times'].append(end_time - start_time)
            algo_results['solutions'].append(best_solution)
            algo_results['history'].append(history)
        
        # Calcula a taxa de sucesso (soluções válidas)
        valid_solutions = sum(1 for s in algo_results['solutions'] 
                            if problem.is_valid_solution(s))
        algo_results['success_rate'] = valid_solutions / n_runs
        
        # Calcula o histórico médio
        algo_results['avg_history'] = np.mean(algo_results['history'], axis=0)
        
        results[algo_name] = algo_results
    
    return results

def main():
    # Configurações do experimento
    n_items = 50
    n_runs = 30
    max_iterations = 100
    
    # Gera uma instância do problema
    problem = generate_random_problem(n_items)
    print(f"\nProblema gerado com {n_items} itens")
    print(f"Capacidade da mochila: {problem.capacity:.2f}")
    print(f"Valor ótimo (programação dinâmica): {problem.get_best_possible_value():.2f}")
    
    # Executa os experimentos
    results = run_experiment(problem, n_runs, max_iterations)
    
    # Cria visualizações
    plot_convergence(
        {algo: results[algo]['avg_history'] for algo in results},
        save_path='convergence.png'
    )
    
    plot_boxplot(
        {algo: results[algo]['values'] for algo in results},
        save_path='boxplot.png'
    )
    
    plot_solution_distribution(
        {algo: results[algo]['solutions'] for algo in results},
        n_items,
        save_path='solution_distribution.png'
    )
    
    # Cria e salva a tabela comparativa
    comparison_table = create_comparison_table(results)
    comparison_table.to_csv('comparison_results.csv', index=False)
    print("\nResultados salvos em 'comparison_results.csv'")
    
    # Exibe os resultados
    print("\nResultados dos experimentos:")
    print(comparison_table.to_string(index=False))

if __name__ == "__main__":
    main() 