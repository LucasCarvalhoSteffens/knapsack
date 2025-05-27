from typing import Dict, Type
from .base import BioInspiredAlgorithm
from utils.problem import KnapsackProblem

class AlgorithmCatalog:
    """Cataloga e fornece instâncias de algoritmos bio-inspirados."""
    def __init__(self):
        self._algorithms: Dict[str, Type[BioInspiredAlgorithm]] = {}

    def register_algorithm(self, name: str, algorithm_class: Type[BioInspiredAlgorithm]):
        """Registra uma classe de algoritmo no catálogo.

        Args:
            name: Nome curto para identificar o algoritmo (ex: 'GA', 'ACO').
            algorithm_class: A classe do algoritmo (deve herdar de BioInspiredAlgorithm).
        """
        if not issubclass(algorithm_class, BioInspiredAlgorithm):
            raise TypeError(f"{algorithm_class.__name__} não é uma subclasse de BioInspiredAlgorithm")
        self._algorithms[name] = algorithm_class

    def get_algorithm(self, name: str, problem: KnapsackProblem, **kwargs) -> BioInspiredAlgorithm:
        """Obtém uma instância de um algoritmo registrado, passando o problema e outros parâmetros.

        Args:
            name: Nome do algoritmo registrado.
            problem: A instância do problema a ser resolvida.
            **kwargs: Argumentos adicionais para o construtor do algoritmo.

        Returns:
            Uma instância do algoritmo solicitado.

        Raises:
            KeyError: Se o nome do algoritmo não for encontrado.
        """
        if name not in self._algorithms:
            raise KeyError(f"Algoritmo '{name}' não encontrado no catálogo.")
        
        algorithm_class = self._algorithms[name]
        return algorithm_class(problem=problem, **kwargs)

    def list_algorithms(self) -> list[str]:
        """Lista os nomes de todos os algoritmos registrados."""
        return list(self._algorithms.keys())

# Instância global do catálogo (pode ser ajustado conforme a necessidade)
catalog = AlgorithmCatalog()

