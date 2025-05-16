import numpy as np
from typing import Tuple, List
from .base import BioInspiredAlgorithm

class AntColony(BioInspiredAlgorithm):
    def __init__(self, problem, n_ants: int = 50,
                 alpha: float = 1.0,  # Importância do feromônio
                 beta: float = 2.0,   # Importância da heurística
                 rho: float = 0.1,    # Taxa de evaporação
                 q0: float = 0.9):    # Probabilidade de escolha gulosa
        """
        Inicializa o Algoritmo de Colônia de Formigas.
        
        Args:
            problem: Instância do problema da mochila
            n_ants: Número de formigas
            alpha: Importância do feromônio
            beta: Importância da heurística
            rho: Taxa de evaporação
            q0: Probabilidade de escolha gulosa
        """
        super().__init__(problem)
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        
        # Inicializa matriz de feromônio
        self.pheromone = np.ones(self.problem.n_items)
        
        # Calcula heurística (razão valor/peso)
        self.heuristic = self.problem.value_weight_ratio
        self.heuristic = self.heuristic / np.max(self.heuristic)  # Normaliza
    
    def initialize_population(self) -> None:
        """
        Inicializa o estado do algoritmo.
        """
        # Reinicia a melhor solução
        self.best_solution = None
        self.best_value = float('-inf')
    
    def construct_solution(self) -> np.ndarray:
        """
        Constrói uma solução usando uma formiga.
        
        Returns:
            Solução construída
        """
        solution = np.zeros(self.problem.n_items)
        remaining_capacity = self.problem.capacity
        
        # Ordena itens por probabilidade
        probabilities = (self.pheromone ** self.alpha) * (self.heuristic ** self.beta)
        probabilities = probabilities / np.sum(probabilities)
        
        # Constrói solução
        while True:
            # Encontra itens que ainda cabem na mochila
            valid_items = np.where(
                (self.problem.weights <= remaining_capacity) & 
                (solution == 0)
            )[0]
            
            if len(valid_items) == 0:
                break
            
            # Escolhe próximo item
            if np.random.random() < self.q0:
                # Escolha gulosa
                item = valid_items[np.argmax(probabilities[valid_items])]
            else:
                # Escolha probabilística
                probs = probabilities[valid_items]
                probs = probs / np.sum(probs)
                item = np.random.choice(valid_items, p=probs)
            
            # Adiciona item à solução
            solution[item] = 1
            remaining_capacity -= self.problem.weights[item]
        
        return solution
    
    def update_pheromone(self, solutions: List[np.ndarray], values: List[float]) -> None:
        """
        Atualiza a matriz de feromônio.
        
        Args:
            solutions: Lista de soluções
            values: Lista de valores das soluções
        """
        # Evaporação
        self.pheromone *= (1 - self.rho)
        
        # Atualização
        for solution, value in zip(solutions, values):
            if value > 0:  # Só atualiza se a solução for válida
                delta = value / np.max(values)  # Normaliza
                self.pheromone += delta * solution
    
    def update_state(self) -> None:
        """
        Atualiza o estado do algoritmo (uma iteração).
        """
        # Constrói soluções
        solutions = []
        values = []
        
        for _ in range(self.n_ants):
            solution = self.construct_solution()
            value = self.evaluate_solution(solution)
            
            solutions.append(solution)
            values.append(value)
            
            # Atualiza melhor solução
            if value > self.best_value:
                self.best_solution = solution.copy()
                self.best_value = value
        
        # Atualiza feromônio
        self.update_pheromone(solutions, values) 