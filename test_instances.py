import numpy as np
import pytest
import os

# Tentativa de organizar a estrutura de pastas para execução dos testes
try:
    if not os.path.exists('algorithms'): os.makedirs('algorithms')
    if not os.path.exists('utils'): os.makedirs('utils')
    # Mover arquivos se estiverem na raiz (ignora erros se já movidos)
    try:
        if os.path.exists('base.py'): os.rename('base.py', 'algorithms/base.py')
        if os.path.exists('genetic_algorithm.py'): os.rename('genetic_algorithm.py', 'algorithms/genetic_algorithm.py')
        if os.path.exists('ant_colony.py'): os.rename('ant_colony.py', 'algorithms/ant_colony.py')
        if os.path.exists('particle_swarm.py'): os.rename('particle_swarm.py', 'algorithms/particle_swarm.py')
        if os.path.exists('cuckoo_search.py'): os.rename('cuckoo_search.py', 'algorithms/cuckoo_search.py')
        # Adicione aqui a movimentação dos arquivos de utils se necessário
        # Ex: if os.path.exists('problem.py'): os.rename('problem.py', 'utils/problem.py')
        # Ex: if os.path.exists('visualization.py'): os.rename('visualization.py', 'utils/visualization.py')
    except OSError as e:
        print(f"Aviso: Erro ao mover arquivos para estrutura de pastas (pode já estar correto): {e}")
    # Criar __init__.py se não existirem
    open('algorithms/__init__.py', 'a').close()
    open('utils/__init__.py', 'a').close()
except Exception as e:
    print(f"Erro crítico ao configurar estrutura de pastas: {e}")

from utils.problem import KnapsackProblem # Assumindo que problem.py está em utils/
from catalog import catalog # Importa a instância do catálogo

# Importar classes apenas para registro (se não feito em main.py ou outro ponto)
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.ant_colony import AntColony
from algorithms.particle_swarm import ParticleSwarm
from algorithms.cuckoo_search import CuckooSearch

# Registrar algoritmos no catálogo (garante que estejam disponíveis para os testes)
# Pode ser redundante se main.py for executado antes, mas torna os testes independentes
if not catalog.list_algorithms(): # Registra apenas se o catálogo estiver vazio
    catalog.register_algorithm("GA", GeneticAlgorithm)
    catalog.register_algorithm("ACO", AntColony)
    catalog.register_algorithm("PSO", ParticleSwarm)
    catalog.register_algorithm("CS", CuckooSearch)

# --- Funções de criação de instâncias (inalteradas) ---

def create_simple_instance():
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 5
    # Assumindo que KnapsackProblem está em utils/problem.py
    return KnapsackProblem(weights, values, capacity)

def create_medium_instance():
    weights = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    values = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    capacity = 200
    return KnapsackProblem(weights, values, capacity)

def create_large_instance():
    n_items = 100
    weights = np.random.uniform(1, 100, n_items)
    values = np.random.uniform(1, 100, n_items)
    capacity = np.sum(weights) * 0.5
    return KnapsackProblem(weights, values, capacity)

# --- Testes refatorados para usar o catálogo ---

@pytest.mark.parametrize("instance_creator, max_iter, min_optimal_ratio", [
    (create_simple_instance, 50, 0.9),
    (create_medium_instance, 100, 0.8),
    (create_large_instance, 200, 0.7),
])
def test_instances_with_catalog(instance_creator, max_iter, min_optimal_ratio):
    """Testa os algoritmos do catálogo em diferentes instâncias."""
    problem = instance_creator()
    optimal_value = problem.get_best_possible_value()
    algorithm_names = catalog.list_algorithms()
    
    print(f"\nTestando instância: {instance_creator.__name__} com {algorithm_names}")
    
    for algo_name in algorithm_names:
        print(f"  Executando {algo_name}...")
        algo = catalog.get_algorithm(algo_name, problem)
        solution, value, _ = algo.run(max_iterations=max_iter)
        
        print(f"    {algo_name} - Solução: {solution}, Valor: {value:.2f} (Ótimo: {optimal_value:.2f})")
        
        # Verifica se a solução é válida
        assert problem.is_valid_solution(solution), f"{algo_name} gerou solução inválida."
        
        # Verifica se o valor está próximo do ótimo
        # Adiciona uma pequena tolerância para evitar falhas por arredondamento
        assert value >= optimal_value * min_optimal_ratio - 1e-9, \
               f"{algo_name} não atingiu {min_optimal_ratio*100}% do ótimo ({optimal_value}). Valor obtido: {value}"

def test_algorithm_parameters():
    """Testa diferentes parâmetros dos algoritmos usando o catálogo."""
    problem = create_medium_instance()
    
    # Testa diferentes tamanhos de população para GA
    for pop_size in [20, 50, 100]:
        print(f"  Testando GA com population_size={pop_size}")
        ga = catalog.get_algorithm("GA", problem, population_size=pop_size)
        solution, value, _ = ga.run(max_iterations=50)
        assert problem.is_valid_solution(solution)
    
    # Testa diferentes taxas de mutação para GA
    for mut_rate in [0.1, 0.2, 0.3]:
        print(f"  Testando GA com mutation_rate={mut_rate}")
        ga = catalog.get_algorithm("GA", problem, mutation_rate=mut_rate)
        solution, value, _ = ga.run(max_iterations=50)
        assert problem.is_valid_solution(solution)
    
    # Testa diferentes números de formigas para ACO
    for n_ants in [20, 50, 100]:
        print(f"  Testando ACO com n_ants={n_ants}")
        aco = catalog.get_algorithm("ACO", problem, n_ants=n_ants)
        solution, value, _ = aco.run(max_iterations=50)
        assert problem.is_valid_solution(solution)
    
    # Adicionar testes para PSO e CS se eles tiverem parâmetros configuráveis importantes
    # Exemplo para PSO:
    # for w_inertia in [0.5, 0.7, 0.9]:
    #     print(f"  Testando PSO com w={w_inertia}")
    #     pso = catalog.get_algorithm("PSO", problem, w=w_inertia)
    #     solution, value, _ = pso.run(max_iterations=50)
    #     assert problem.is_valid_solution(solution)

def test_convergence():
    """Testa a convergência dos algoritmos do catálogo."""
    problem = create_medium_instance()
    algorithm_names = catalog.list_algorithms()
    max_iterations = 100
    
    print(f"\nTestando convergência com {algorithm_names}")
    
    for algo_name in algorithm_names:
        print(f"  Verificando convergência de {algo_name}...")
        algo = catalog.get_algorithm(algo_name, problem)
        _, _, history = algo.run(max_iterations=max_iterations)
        
        # Verifica se o histórico tem o tamanho correto
        assert len(history) == max_iterations, f"{algo_name} - Histórico com tamanho incorreto."
        
        # Verifica se o valor melhora ou permanece igual (não decresce)
        for i in range(1, len(history)):
            assert history[i] >= history[i-1] - 1e-9, \
                   f"{algo_name} - Valor decresceu no histórico na iteração {i}."

if __name__ == "__main__":
    # Executa os testes usando pytest
    # Garante que a estrutura de pastas esteja correta antes de rodar
    print("Executando testes com pytest...")
    pytest.main(["-v", __file__]) # -v para verbose output

