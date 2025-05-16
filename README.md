# Solução do Problema da Mochila usando Algoritmos Bio-inspirados

Este projeto implementa e compara diferentes algoritmos bio-inspirados para resolver o Problema da Mochila 0/1. A implementação inclui quatro algoritmos distintos: Algoritmo Genético (GA), Colônia de Formigas (ACO), Otimização por Enxame de Partículas (PSO) e Algoritmo do Cuckoo (CS).

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Algoritmos Implementados](#algoritmos-implementados)
- [Instalação](#instalação)
- [Uso](#uso)
- [Análise de Resultados](#análise-de-resultados)
- [Testes](#testes)
- [Contribuindo](#contribuindo)
- [Licença](#licença)

## 🎯 Visão Geral

O Problema da Mochila 0/1 é um problema clássico de otimização combinatória onde, dado um conjunto de itens com pesos e valores, o objetivo é selecionar um subconjunto de itens que maximize o valor total sem exceder uma capacidade máxima.

Este projeto implementa soluções usando algoritmos bio-inspirados, que são meta-heurísticas baseadas em comportamentos naturais. Cada algoritmo tem suas próprias características e parâmetros que podem ser ajustados para otimizar o desempenho.

## 📁 Estrutura do Projeto

```
knapsack/
├── algorithms/
│   ├── __init__.py
│   ├── base.py           # Classe base abstrata
│   ├── genetic_algorithm.py
│   ├── ant_colony.py
│   ├── particle_swarm.py
│   └── cuckoo_search.py
├── utils/
│   ├── __init__.py
│   ├── problem.py        # Definição do problema
│   └── visualization.py  # Ferramentas de visualização
├── tests/
│   ├── __init__.py
│   └── test_instances.py # Casos de teste
├── main.py              # Script principal
└── requirements.txt     # Dependências
```

## 🧬 Algoritmos Implementados

### 1. Algoritmo Genético (GA)
- **Representação**: Soluções binárias
- **Operadores**:
  - Seleção por torneio
  - Crossover de um ponto
  - Mutação bit-flip
- **Parâmetros**:
  - `population_size`: Tamanho da população
  - `mutation_rate`: Taxa de mutação
  - `elite_size`: Número de melhores indivíduos preservados

### 2. Colônia de Formigas (ACO)
- **Mecanismo**: Construção baseada em feromônio
- **Heurística**: Razão valor/peso
- **Parâmetros**:
  - `n_ants`: Número de formigas
  - `alpha`: Importância do feromônio
  - `beta`: Importância da heurística
  - `rho`: Taxa de evaporação
  - `q0`: Probabilidade de escolha gulosa

### 3. Otimização por Enxame de Partículas (PSO)
- **Representação**: Partículas com posições binárias
- **Mecanismo**: Movimento baseado em melhor posição individual e global
- **Parâmetros**:
  - `n_particles`: Número de partículas
  - `w`: Peso da inércia
  - `c1`: Coeficiente cognitivo
  - `c2`: Coeficiente social

### 4. Algoritmo do Cuckoo (CS)
- **Mecanismo**: Voo de Lévy e substituição de ninhos
- **Parâmetros**:
  - `n_nests`: Número de ninhos
  - `pa`: Probabilidade de abandono
  - `beta`: Parâmetro do voo de Lévy

## 💻 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/LucasCarvalhoSteffens/knapsack.git
cd knapsack
```

2. Crie um ambiente virtual (opcional, mas recomendado):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 🚀 Uso

### Executando Experimentos

O script principal (`main.py`) executa experimentos comparativos entre os algoritmos:

```bash
python main.py
```

Isso irá:
1. Gerar uma instância aleatória do problema
2. Executar cada algoritmo múltiplas vezes
3. Gerar visualizações comparativas
4. Salvar resultados em CSV

### Personalizando Experimentos

Você pode modificar os parâmetros no `main.py`:

```python
# Configurações do experimento
n_items = 50        # Número de itens
n_runs = 30         # Número de execuções
max_iterations = 100 # Iterações por execução
```

### Usando os Algoritmos Individualmente

```python
from utils.problem import KnapsackProblem
from algorithms.genetic_algorithm import GeneticAlgorithm

# Cria uma instância do problema
problem = KnapsackProblem(weights=[2, 3, 4, 5], 
                         values=[3, 4, 5, 6], 
                         capacity=5)

# Cria e executa o algoritmo
ga = GeneticAlgorithm(problem)
solution, value, history = ga.run(max_iterations=100)
```

## 📊 Análise de Resultados

O projeto gera várias visualizações:

1. **Convergência** (`convergence.png`):
   - Curvas de convergência para cada algoritmo
   - Comparação da evolução do valor ao longo das iterações

2. **Boxplot** (`boxplot.png`):
   - Distribuição dos resultados
   - Comparação da robustez dos algoritmos

3. **Distribuição de Soluções** (`solution_distribution.png`):
   - Frequência de seleção de cada item
   - Padrões de seleção dos algoritmos

4. **Tabela Comparativa** (`comparison_results.csv`):
   - Métricas detalhadas para cada algoritmo
   - Tempo de execução, valores médios, etc.

## 🧪 Testes

Execute os testes unitários:

```bash
pytest tests/
```

Os testes incluem:
- Instâncias de diferentes tamanhos
- Verificação de validade das soluções
- Testes de convergência
- Testes de parâmetros

## 🤝 Contribuindo

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 📚 Referências

- [Algoritmos Bio-inspirados](https://en.wikipedia.org/wiki/Bio-inspired_computing)
- [Problema da Mochila](https://en.wikipedia.org/wiki/Knapsack_problem)
- [Otimização por Enxame de Partículas](https://en.wikipedia.org/wiki/Particle_swarm_optimization)
- [Algoritmo do Cuckoo](https://en.wikipedia.org/wiki/Cuckoo_search) 