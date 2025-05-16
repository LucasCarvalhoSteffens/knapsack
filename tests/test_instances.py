import numpy as np
import pytest
from utils.problem import KnapsackProblem
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.ant_colony import AntColony
from algorithms.particle_swarm import ParticleSwarm
from algorithms.cuckoo_search import CuckooSearch

# Instância de teste simples
def create_simple_instance():
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 5
    return KnapsackProblem(weights, values, capacity)

# Instância de teste média
def create_medium_instance():
    weights = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    values = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    capacity = 200
    return KnapsackProblem(weights, values, capacity)

# Instância de teste grande
def create_large_instance():
    n_items = 100
    weights = np.random.uniform(1, 100, n_items)
    values = np.random.uniform(1, 100, n_items)
    capacity = np.sum(weights) * 0.5  # 50% da capacidade total
    return KnapsackProblem(weights, values, capacity)

def test_simple_instance():
    """Testa os algoritmos em uma instância simples."""
    problem = create_simple_instance()
    optimal_value = problem.get_best_possible_value()
    
    algorithms = [
        GeneticAlgorithm(problem),
        AntColony(problem),
        ParticleSwarm(problem),
        CuckooSearch(problem)
    ]
    
    for algo in algorithms:
        solution, value, _ = algo.run(max_iterations=50)
        
        # Verifica se a solução é válida
        assert problem.is_valid_solution(solution)
        
        # Verifica se o valor está próximo do ótimo
        assert value >= optimal_value * 0.9  # 90% do ótimo

def test_medium_instance():
    """Testa os algoritmos em uma instância média."""
    problem = create_medium_instance()
    optimal_value = problem.get_best_possible_value()
    
    algorithms = [
        GeneticAlgorithm(problem),
        AntColony(problem),
        ParticleSwarm(problem),
        CuckooSearch(problem)
    ]
    
    for algo in algorithms:
        solution, value, _ = algo.run(max_iterations=100)
        
        # Verifica se a solução é válida
        assert problem.is_valid_solution(solution)
        
        # Verifica se o valor está próximo do ótimo
        assert value >= optimal_value * 0.8  # 80% do ótimo

def test_large_instance():
    """Testa os algoritmos em uma instância grande."""
    problem = create_large_instance()
    optimal_value = problem.get_best_possible_value()
    
    algorithms = [
        GeneticAlgorithm(problem),
        AntColony(problem),
        ParticleSwarm(problem),
        CuckooSearch(problem)
    ]
    
    for algo in algorithms:
        solution, value, _ = algo.run(max_iterations=200)
        
        # Verifica se a solução é válida
        assert problem.is_valid_solution(solution)
        
        # Verifica se o valor está próximo do ótimo
        assert value >= optimal_value * 0.7  # 70% do ótimo

def test_algorithm_parameters():
    """Testa diferentes parâmetros dos algoritmos."""
    problem = create_medium_instance()
    
    # Testa diferentes tamanhos de população
    for pop_size in [20, 50, 100]:
        ga = GeneticAlgorithm(problem, population_size=pop_size)
        solution, value, _ = ga.run(max_iterations=50)
        assert problem.is_valid_solution(solution)
    
    # Testa diferentes taxas de mutação
    for mut_rate in [0.1, 0.2, 0.3]:
        ga = GeneticAlgorithm(problem, mutation_rate=mut_rate)
        solution, value, _ = ga.run(max_iterations=50)
        assert problem.is_valid_solution(solution)
    
    # Testa diferentes números de formigas
    for n_ants in [20, 50, 100]:
        aco = AntColony(problem, n_ants=n_ants)
        solution, value, _ = aco.run(max_iterations=50)
        assert problem.is_valid_solution(solution)

def test_convergence():
    """Testa a convergência dos algoritmos."""
    problem = create_medium_instance()
    
    algorithms = [
        GeneticAlgorithm(problem),
        AntColony(problem),
        ParticleSwarm(problem),
        CuckooSearch(problem)
    ]
    
    for algo in algorithms:
        _, _, history = algo.run(max_iterations=100)
        
        # Verifica se o histórico tem o tamanho correto
        assert len(history) == 100
        
        # Verifica se o valor melhora ou permanece igual
        for i in range(1, len(history)):
            assert history[i] >= history[i-1]

if __name__ == "__main__":
    pytest.main([__file__]) 