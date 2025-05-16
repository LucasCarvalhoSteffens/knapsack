import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
import pandas as pd

def plot_convergence(history: Dict[str, List[float]], 
                    title: str = "Convergência dos Algoritmos",
                    save_path: str = None):
    """
    Plota a curva de convergência para diferentes algoritmos.
    
    Args:
        history: Dicionário com histórico de valores para cada algoritmo
        title: Título do gráfico
        save_path: Caminho para salvar o gráfico (opcional)
    """
    plt.figure(figsize=(10, 6))
    
    for algo_name, values in history.items():
        plt.plot(values, label=algo_name)
    
    plt.xlabel("Iteração")
    plt.ylabel("Valor Total")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_boxplot(results: Dict[str, List[float]], 
                title: str = "Comparação dos Algoritmos",
                save_path: str = None):
    """
    Cria um boxplot comparando os resultados dos diferentes algoritmos.
    
    Args:
        results: Dicionário com resultados de cada algoritmo
        title: Título do gráfico
        save_path: Caminho para salvar o gráfico (opcional)
    """
    plt.figure(figsize=(10, 6))
    
    data = [values for values in results.values()]
    labels = list(results.keys())
    
    plt.boxplot(data, labels=labels)
    plt.ylabel("Valor Total")
    plt.title(title)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def create_comparison_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Cria uma tabela comparativa com métricas dos algoritmos.
    
    Args:
        results: Dicionário com resultados detalhados de cada algoritmo
        
    Returns:
        DataFrame com as métricas comparativas
    """
    metrics = []
    
    for algo_name, algo_results in results.items():
        metrics.append({
            'Algoritmo': algo_name,
            'Melhor Valor': np.max(algo_results['values']),
            'Valor Médio': np.mean(algo_results['values']),
            'Desvio Padrão': np.std(algo_results['values']),
            'Tempo Médio (s)': np.mean(algo_results['times']),
            'Taxa de Sucesso': algo_results['success_rate']
        })
    
    return pd.DataFrame(metrics)

def plot_solution_distribution(solutions: Dict[str, List[np.ndarray]], 
                             n_items: int,
                             title: str = "Distribuição de Itens Selecionados",
                             save_path: str = None):
    """
    Plota a distribuição de itens selecionados para cada algoritmo.
    
    Args:
        solutions: Dicionário com soluções de cada algoritmo
        n_items: Número total de itens
        title: Título do gráfico
        save_path: Caminho para salvar o gráfico (opcional)
    """
    plt.figure(figsize=(12, 6))
    
    x = np.arange(n_items)
    width = 0.8 / len(solutions)
    
    for i, (algo_name, algo_solutions) in enumerate(solutions.items()):
        # Calcula a frequência de seleção de cada item
        item_freq = np.mean(algo_solutions, axis=0)
        plt.bar(x + i*width, item_freq, width, label=algo_name)
    
    plt.xlabel("Índice do Item")
    plt.ylabel("Frequência de Seleção")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.close() 