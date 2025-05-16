import numpy as np
from typing import Tuple, List
from .base import BioInspiredAlgorithm

class GeneticAlgorithm(BioInspiredAlgorithm):
    def __init__(self, problem, population_size: int = 100,
                 mutation_rate: float = 0.1,
                 elite_size: int = 10):
        """
        Inicializa o Algoritmo Genético.
        
        Args:
            problem: Instância do problema da mochila
            population_size: Tamanho da população
            mutation_rate: Taxa de mutação
            elite_size: Número de melhores indivíduos preservados
        """
        super().__init__(problem)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.population = None
        self.fitness = None
    
    def initialize_population(self) -> None:
        """
        Inicializa a população com soluções aleatórias.
        """
        # Gera população inicial aleatória
        self.population = np.random.randint(0, 2, 
                                          (self.population_size, self.problem.n_items))
        
        # Repara soluções inválidas
        for i in range(self.population_size):
            self.population[i] = self.repair_solution(self.population[i])
        
        # Calcula fitness inicial
        self.fitness = np.array([self.evaluate_solution(sol) for sol in self.population])
        
        # Atualiza melhor solução
        best_idx = np.argmax(self.fitness)
        self.best_solution = self.population[best_idx].copy()
        self.best_value = self.fitness[best_idx]
    
    def select_parents(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Seleciona dois pais usando seleção por torneio.
        
        Returns:
            Tuple com dois pais selecionados
        """
        # Seleção por torneio
        def tournament_select():
            candidates = np.random.choice(self.population_size, 3, replace=False)
            winner = candidates[np.argmax(self.fitness[candidates])]
            return self.population[winner]
        
        return tournament_select(), tournament_select()
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realiza o crossover entre dois pais.
        
        Args:
            parent1: Primeiro pai
            parent2: Segundo pai
            
        Returns:
            Tuple com dois filhos
        """
        # Crossover de um ponto
        point = np.random.randint(1, self.problem.n_items)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        
        return child1, child2
    
    def mutate(self, solution: np.ndarray) -> np.ndarray:
        """
        Aplica mutação em uma solução.
        
        Args:
            solution: Solução a ser mutada
            
        Returns:
            Solução mutada
        """
        # Mutação bit-flip
        mask = np.random.random(self.problem.n_items) < self.mutation_rate
        solution[mask] = 1 - solution[mask]
        return solution
    
    def update_state(self) -> None:
        """
        Atualiza o estado do algoritmo (uma geração).
        """
        new_population = []
        
        # Preserva os melhores indivíduos (elitismo)
        elite_indices = np.argsort(self.fitness)[-self.elite_size:]
        new_population.extend(self.population[elite_indices])
        
        # Gera o resto da população
        while len(new_population) < self.population_size:
            # Seleciona pais
            parent1, parent2 = self.select_parents()
            
            # Realiza crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Aplica mutação
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            # Repara soluções inválidas
            child1 = self.repair_solution(child1)
            child2 = self.repair_solution(child2)
            
            new_population.extend([child1, child2])
        
        # Atualiza população
        self.population = np.array(new_population[:self.population_size])
        
        # Atualiza fitness
        self.fitness = np.array([self.evaluate_solution(sol) for sol in self.population])
        
        # Atualiza melhor solução
        best_idx = np.argmax(self.fitness)
        if self.fitness[best_idx] > self.best_value:
            self.best_solution = self.population[best_idx].copy()
            self.best_value = self.fitness[best_idx] 