import numpy as np
import math
from typing import Tuple, List
from .base import BioInspiredAlgorithm

class CuckooSearch(BioInspiredAlgorithm):
    def __init__(self, problem, n_nests: int = 50,
                 pa: float = 0.25,  # Probabilidade de abandono
                 beta: float = 1.5):  # Parâmetro do voo de Lévy
        """
        Inicializa o Algoritmo do Cuckoo.
        
        Args:
            problem: Instância do problema da mochila
            n_nests: Número de ninhos
            pa: Probabilidade de abandono
            beta: Parâmetro do voo de Lévy
        """
        super().__init__(problem)
        self.n_nests = n_nests
        self.pa = pa
        self.beta = beta
        self.nests = None
        self.values = None
    
    def initialize_population(self) -> None:
        """
        Inicializa a população de ninhos.
        """
        # Inicializa ninhos aleatórios
        self.nests = np.random.randint(0, 2, 
                                     (self.n_nests, self.problem.n_items))
        
        # Repara soluções inválidas
        for i in range(self.n_nests):
            self.nests[i] = self.repair_solution(self.nests[i])
        
        # Avalia ninhos
        self.values = np.array([self.evaluate_solution(nest) 
                              for nest in self.nests])
        
        # Atualiza melhor solução
        best_idx = np.argmax(self.values)
        self.best_solution = self.nests[best_idx].copy()
        self.best_value = self.values[best_idx]
    
    def levy_flight(self, size: int) -> np.ndarray:
        """
        Gera um passo de voo de Lévy.
        
        Args:
            size: Tamanho do passo
            
        Returns:
            Array com o passo de Lévy
        """
        # Gera números aleatórios com distribuição de Lévy
        sigma = (math.gamma(1 + self.beta) * 
                math.sin(math.pi * self.beta / 2) / 
                (math.gamma((1 + self.beta) / 2) * 
                 self.beta * 2 ** ((self.beta - 1) / 2))) ** (1 / self.beta)
        
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        
        return u / (np.abs(v) ** (1 / self.beta))
    
    def update_state(self) -> None:
        """
        Atualiza o estado do algoritmo (uma iteração).
        """
        # Gera novos ninhos usando voo de Lévy
        for i in range(self.n_nests):
            # Seleciona um ninho aleatório
            j = np.random.randint(0, self.n_nests)
            
            # Gera novo ninho
            step = self.levy_flight(self.problem.n_items)
            new_nest = self.nests[i].copy()
            
            # Aplica o passo de Lévy
            mask = np.random.random(self.problem.n_items) < np.abs(step)
            new_nest[mask] = 1 - new_nest[mask]  # Inverte bits selecionados
            
            # Repara solução inválida
            new_nest = self.repair_solution(new_nest)
            
            # Avalia novo ninho
            new_value = self.evaluate_solution(new_nest)
            
            # Substitui se for melhor
            if new_value > self.values[i]:
                self.nests[i] = new_nest
                self.values[i] = new_value
        
        # Abandona piores ninhos
        worst_nests = np.argsort(self.values)[:int(self.n_nests * self.pa)]
        for i in worst_nests:
            # Gera novo ninho aleatório
            self.nests[i] = np.random.randint(0, 2, self.problem.n_items)
            self.nests[i] = self.repair_solution(self.nests[i])
            self.values[i] = self.evaluate_solution(self.nests[i])
        
        # Atualiza melhor solução
        best_idx = np.argmax(self.values)
        if self.values[best_idx] > self.best_value:
            self.best_solution = self.nests[best_idx].copy()
            self.best_value = self.values[best_idx] 