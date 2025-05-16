from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List

from utils.problem import KnapsackProblem

class BioInspiredAlgorithm(ABC):
    def __init__(self, problem: KnapsackProblem):
        """
        Inicializa o algoritmo bio-inspirado.
        
        Args:
            problem: Instância do problema da mochila
        """
        self.problem = problem
        self.best_solution = None
        self.best_value = float('-inf')
        self.history = []
    
    @abstractmethod
    def initialize_population(self) -> None:
        """
        Inicializa a população/estado inicial do algoritmo.
        """
        pass
    
    @abstractmethod
    def update_state(self) -> None:
        """
        Atualiza o estado do algoritmo (uma iteração).
        """
        pass
    
    def evaluate_solution(self, solution: np.ndarray) -> float:
        """
        Avalia uma solução do problema.
        
        Args:
            solution: Array binário representando os itens selecionados
            
        Returns:
            Valor total da solução (0 se inválida)
        """
        value, _ = self.problem.evaluate_solution(solution)
        return value
    
    def repair_solution(self, solution: np.ndarray) -> np.ndarray:
        """
        Repara uma solução inválida removendo itens até que a capacidade seja respeitada.
        
        Args:
            solution: Array binário representando os itens selecionados
            
        Returns:
            Solução reparada
        """
        if self.problem.is_valid_solution(solution):
            return solution
        
        # Cria uma cópia da solução
        repaired = solution.copy()
        
        # Enquanto a solução for inválida
        while not self.problem.is_valid_solution(repaired):
            # Encontra os itens selecionados
            selected = np.where(repaired == 1)[0]
            
            if len(selected) == 0:
                break
            
            # Calcula a razão valor/peso para cada item selecionado
            ratios = self.problem.value_weight_ratio[selected]
            
            # Remove o item com menor razão valor/peso
            worst_item = selected[np.argmin(ratios)]
            repaired[worst_item] = 0
        
        return repaired
    
    def run(self, max_iterations: int) -> Tuple[np.ndarray, float, List[float]]:
        """
        Executa o algoritmo por um número máximo de iterações.
        
        Args:
            max_iterations: Número máximo de iterações
            
        Returns:
            Tuple contendo (melhor_solução, melhor_valor, histórico)
        """
        self.initialize_population()
        self.history = []
        
        for _ in range(max_iterations):
            self.update_state()
            self.history.append(self.best_value)
        
        return self.best_solution, self.best_value, self.history 