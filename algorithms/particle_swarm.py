import numpy as np
from typing import Tuple, List
from .base import BioInspiredAlgorithm

class ParticleSwarm(BioInspiredAlgorithm):
    def __init__(self, problem, n_particles: int = 50,
                 w: float = 0.7,    # Inércia
                 c1: float = 1.5,   # Coeficiente cognitivo
                 c2: float = 1.5):  # Coeficiente social
        """
        Inicializa o Algoritmo de Otimização por Enxame de Partículas.
        
        Args:
            problem: Instância do problema da mochila
            n_particles: Número de partículas
            w: Peso da inércia
            c1: Coeficiente cognitivo
            c2: Coeficiente social
        """
        super().__init__(problem)
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Estado das partículas
        self.positions = None  # Posições atuais
        self.velocities = None  # Velocidades
        self.pbest = None  # Melhor posição individual
        self.pbest_values = None  # Valores das melhores posições individuais
        self.gbest = None  # Melhor posição global
        self.gbest_value = float('-inf')
    
    def initialize_population(self) -> None:
        """
        Inicializa a população de partículas.
        """
        # Inicializa posições aleatórias
        self.positions = np.random.random((self.n_particles, self.problem.n_items))
        self.positions = (self.positions > 0.5).astype(int)  # Binariza
        
        # Repara soluções inválidas
        for i in range(self.n_particles):
            self.positions[i] = self.repair_solution(self.positions[i])
        
        # Inicializa velocidades
        self.velocities = np.random.uniform(-1, 1, 
                                          (self.n_particles, self.problem.n_items))
        
        # Inicializa melhores posições individuais
        self.pbest = self.positions.copy()
        self.pbest_values = np.array([self.evaluate_solution(sol) 
                                    for sol in self.positions])
        
        # Atualiza melhor posição global
        best_idx = np.argmax(self.pbest_values)
        self.gbest = self.pbest[best_idx].copy()
        self.gbest_value = self.pbest_values[best_idx]
        
        # Atualiza melhor solução global
        self.best_solution = self.gbest.copy()
        self.best_value = self.gbest_value
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Aplica a função sigmoid para binarização.
        
        Args:
            x: Array de entrada
            
        Returns:
            Array com valores entre 0 e 1
        """
        return 1 / (1 + np.exp(-x))
    
    def update_state(self) -> None:
        """
        Atualiza o estado do algoritmo (uma iteração).
        """
        # Atualiza velocidades e posições
        for i in range(self.n_particles):
            # Atualiza velocidade
            r1, r2 = np.random.random(2)
            cognitive = self.c1 * r1 * (self.pbest[i] - self.positions[i])
            social = self.c2 * r2 * (self.gbest - self.positions[i])
            self.velocities[i] = (self.w * self.velocities[i] + 
                                cognitive + social)
            
            # Atualiza posição
            sigmoid_vel = self.sigmoid(self.velocities[i])
            self.positions[i] = (np.random.random(self.problem.n_items) < sigmoid_vel).astype(int)
            
            # Repara solução inválida
            self.positions[i] = self.repair_solution(self.positions[i])
            
            # Avalia nova posição
            value = self.evaluate_solution(self.positions[i])
            
            # Atualiza melhor posição individual
            if value > self.pbest_values[i]:
                self.pbest[i] = self.positions[i].copy()
                self.pbest_values[i] = value
                
                # Atualiza melhor posição global
                if value > self.gbest_value:
                    self.gbest = self.positions[i].copy()
                    self.gbest_value = value
                    
                    # Atualiza melhor solução global
                    self.best_solution = self.gbest.copy()
                    self.best_value = self.gbest_value 