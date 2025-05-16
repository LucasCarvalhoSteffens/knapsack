import numpy as np
from typing import List, Tuple, Optional

class KnapsackProblem:
    def __init__(self, weights: List[float], values: List[float], capacity: float):
        """
        Inicializa uma instância do problema da mochila.
        
        Args:
            weights: Lista de pesos dos itens
            values: Lista de valores dos itens
            capacity: Capacidade máxima da mochila
        """
        self.weights = np.array(weights)
        self.values = np.array(values)
        self.capacity = capacity
        self.n_items = len(weights)
        
        # Calcula a razão valor/peso para heurísticas
        self.value_weight_ratio = self.values / self.weights
        
    def evaluate_solution(self, solution: np.ndarray) -> Tuple[float, float]:
        """
        Avalia uma solução do problema da mochila.
        
        Args:
            solution: Array binário representando os itens selecionados
            
        Returns:
            Tuple contendo (valor_total, peso_total)
        """
        total_value = np.sum(self.values * solution)
        total_weight = np.sum(self.weights * solution)
        
        # Penaliza soluções que excedem a capacidade
        if total_weight > self.capacity:
            total_value = 0
            
        return total_value, total_weight
    
    def is_valid_solution(self, solution: np.ndarray) -> bool:
        """
        Verifica se uma solução é válida (não excede a capacidade).
        
        Args:
            solution: Array binário representando os itens selecionados
            
        Returns:
            True se a solução é válida, False caso contrário
        """
        total_weight = np.sum(self.weights * solution)
        return total_weight <= self.capacity
    
    def get_best_possible_value(self) -> float:
        """
        Retorna o melhor valor possível (solução ótima) para o problema.
        Esta é uma implementação simples usando programação dinâmica.
        
        Returns:
            O valor máximo possível
        """
        dp = np.zeros((self.n_items + 1, int(self.capacity) + 1))
        
        for i in range(1, self.n_items + 1):
            for w in range(int(self.capacity) + 1):
                if self.weights[i-1] <= w:
                    dp[i][w] = max(dp[i-1][w], 
                                 dp[i-1][w-int(self.weights[i-1])] + self.values[i-1])
                else:
                    dp[i][w] = dp[i-1][w]
                    
        return dp[self.n_items][int(self.capacity)]

def generate_random_problem(n_items: int, 
                          max_weight: float = 100.0,
                          max_value: float = 100.0,
                          capacity_ratio: float = 0.5) -> KnapsackProblem:
    """
    Gera uma instância aleatória do problema da mochila.
    
    Args:
        n_items: Número de itens
        max_weight: Peso máximo de um item
        max_value: Valor máximo de um item
        capacity_ratio: Razão entre a capacidade total e a soma dos pesos
        
    Returns:
        Uma instância do problema da mochila
    """
    weights = np.random.uniform(1, max_weight, n_items)
    values = np.random.uniform(1, max_value, n_items)
    total_weight = np.sum(weights)
    capacity = total_weight * capacity_ratio
    
    return KnapsackProblem(weights, values, capacity) 