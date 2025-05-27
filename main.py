import numpy as np
import time
from typing import Dict, List, Any
import pandas as pd
from tqdm import tqdm
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.problem import KnapsackProblem, generate_random_problem
from utils.visualization import (plot_convergence, plot_boxplot, 
                               create_comparison_table, plot_solution_distribution)

# Importar o catálogo e as classes de algoritmo para registro
from catalog import catalog
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.ant_colony import AntColony
from algorithms.particle_swarm import ParticleSwarm
from algorithms.cuckoo_search import CuckooSearch

# Registrar os algoritmos no catálogo
catalog.register_algorithm('GA', GeneticAlgorithm)
catalog.register_algorithm('ACO', AntColony)
catalog.register_algorithm('PSO', ParticleSwarm)
catalog.register_algorithm('CS', CuckooSearch)

def run_experiment(problem: KnapsackProblem,
                  n_runs: int = 30,
                  max_iterations: int = 100) -> Dict[str, Dict[str, Any]]:
    """
    Executa experimentos comparando diferentes algoritmos usando o catálogo.
    
    Args:
        problem: Instância do problema da mochila
        n_runs: Número de execuções para cada algoritmo
        max_iterations: Número máximo de iterações por execução
        
    Returns:
        Dicionário com resultados detalhados de cada algoritmo
    """
    results = {}
    algorithm_names = catalog.list_algorithms()
    
    for algo_name in algorithm_names:
        print(f"\nExecutando {algo_name}...")
        
        algo_results = {
            'values': [],
            'times': [],
            'solutions': [],
            'history': []
        }
        
        for _ in tqdm(range(n_runs)):
            start_time = time.time()
            
            # Obtém uma nova instância do algoritmo do catálogo
            # Passa os parâmetros específicos do algoritmo, se houver
            # Exemplo: Se GA precisasse de pop_size, seria catalog.get_algorithm('GA', problem, population_size=100)
            algorithm = catalog.get_algorithm(algo_name, problem)
            
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
    # Cria as pastas necessárias se não existirem
    import os
    os.makedirs('utils', exist_ok=True)
    os.makedirs('algorithms', exist_ok=True)
    # É necessário mover os arquivos para as pastas corretas
    # ou ajustar os imports para refletir a estrutura atual
    # Exemplo: Se base.py está no mesmo nível que catalog.py
    # o import em catalog.py seria from base import BioInspiredAlgorithm
    
    # Ajuste temporário para execução (assumindo arquivos no mesmo nível)
    # Idealmente, a estrutura de pastas deve ser corrigida
    # Ex: Mover base.py, genetic_algorithm.py etc para 'algorithms/'
    # Ex: Mover problem.py, visualization.py para 'utils/'
    
    # Cria arquivos vazios para evitar erros de import iniciais
    # (Isso é um paliativo, a estrutura de pastas deve ser organizada)
    open('utils/__init__.py', 'a').close()
    open('algorithms/__init__.py', 'a').close()
    # Mova os arquivos .py correspondentes para essas pastas
    
    # Tentativa de mover arquivos (pode precisar de ajuste manual)
    try:
        if os.path.exists('base.py'): os.rename('base.py', 'algorithms/base.py')
        if os.path.exists('genetic_algorithm.py'): os.rename('genetic_algorithm.py', 'algorithms/genetic_algorithm.py')
        if os.path.exists('ant_colony.py'): os.rename('ant_colony.py', 'algorithms/ant_colony.py')
        if os.path.exists('particle_swarm.py'): os.rename('particle_swarm.py', 'algorithms/particle_swarm.py')
        if os.path.exists('cuckoo_search.py'): os.rename('cuckoo_search.py', 'algorithms/cuckoo_search.py')
        # Assumindo que problem.py e visualization.py existem e estão no diretório raiz
        # if os.path.exists('problem.py'): os.rename('problem.py', 'utils/problem.py') 
        # if os.path.exists('visualization.py'): os.rename('visualization.py', 'utils/visualization.py')
        pass # Adicione aqui a movimentação dos arquivos de utils se necessário
    except Exception as e:
        print(f"Erro ao organizar arquivos: {e}. Verifique a estrutura de pastas.")

    main()

