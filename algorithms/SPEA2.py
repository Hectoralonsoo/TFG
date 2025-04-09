from inspyred.ec import EvolutionaryComputation
from inspyred.ec import Individual
import numpy as np
from scipy.spatial import distance
from inspyred.ec import selectors, variators, terminators


class SPEA2(EvolutionaryComputation):
    def __init__(self, random):
        super().__init__(random)
        self.archive = []
        self.archive_size = 300
        self.k = 3

        # ✅ Define atributos por defecto
        self.default_selector = selectors.tournament_selection
        self.default_variator = [variators.uniform_crossover, variators.random_reset_mutation]
        self.default_terminator = terminators.generation_termination

    def evolve(self, pop_size, maximize, max_generations, num_selected, **kwargs):
        """
        Implementación personalizada del algoritmo SPEA2.
        """
        self._maximize = maximize
        self.num_generations = 0
        self._num_evaluations = 0
        self.archive = []

        generation = 0
        population = [Individual(self.generator(random=self._random, args=kwargs)) for _ in range(pop_size)]
        population = self._evaluate_population(population, kwargs)

        # Calcular k dinámicamente para densidad
        self.k = kwargs.get('k', max(1, int(len(population) ** 0.5)))

        while generation < max_generations:
            combined = population + self.archive

            self._assign_spea2_fitness(combined)
            self.archive = self._select_archive(combined)

            if self.observer:
                self.observer(self.archive, generation, self._num_evaluations, kwargs)

          #  if self.terminator:
           #     if self.terminator(self.archive, generation, self._num_evaluations, kwargs):
            #        break

            mating_pool = self._select_parents(self.archive, num_selected)
            offspring = []

            while len(offspring) < pop_size:
                selected = self._random.sample(mating_pool, 2)
                parents = [p.candidate for p in selected]
                children = parents
                for v in self.variator:
                    children = v(self._random, children, {'random': self._random, '_ec': self, **kwargs})
                offspring.extend([Individual(child) for child in children])

            offspring = offspring[:pop_size]
            offspring = self._evaluate_population(offspring, kwargs)
            population = offspring

            generation += 1
            self.num_generations = generation
            self._num_evaluations += len(population)

        return self.archive

    def _generate_initial_population(self, **kwargs):
        """Genera la población inicial y crea objetos Individual."""
        pop_size = kwargs.get('pop_size', self._pop_size)
        generator = kwargs.get('generator')
        candidates = generator(random=self._random, args=kwargs)

        population = []
        for candidate in candidates:
            population.append(Individual(candidate))

        return population

    def _evaluate_population(self, population, kwargs):
        """Evalúa la población y asigna valores objetivo."""
        if population:
            candidates = [p.candidate for p in population]
            fitness_values = self.evaluator(candidates=candidates, args=kwargs)

            for index, individual in enumerate(population):
                individual.objective_values = fitness_values[index]
                # No asignamos fitness aquí, se hará en _assign_spea2_fitness

        return population

    def _select_parents(self, archive, num_selected):
        """Selección por torneo binario."""
        selected = []
        if len(archive) == 0:
            return selected

        for _ in range(num_selected):
            competitors = self._random.sample(archive, min(2, len(archive)))
            winner = min(competitors, key=lambda x: x.fitness)
            selected.append(winner)
        return selected

    def _assign_spea2_fitness(self, individuals):
        """
        Asigna fitness según el método SPEA2:
        1. Cálculo de fuerza (S)
        2. Suma de fuerzas (R)
        3. Estimación de densidad (D)
        4. Fitness = R + D
        """
        size = len(individuals)
        if size == 0:
            return

        # Matriz de dominancia: dominance[i][j] = True si i domina a j
        dominance = np.zeros((size, size), dtype=bool)

        # Calcular relaciones de dominancia y contar individuos dominados (S)
        S = np.zeros(size, dtype=int)

        for i in range(size):
            for j in range(size):
                if i != j:
                    if self._dominates(individuals[i], individuals[j]):
                        dominance[i][j] = True
                        S[i] += 1

        # Calcular valor de fuerza raw (R)
        R = np.zeros(size)
        for i in range(size):
            # Suma de fuerzas de todos los individuos que dominan a i
            R[i] = sum(S[j] for j in range(size) if dominance[j][i])

        # Calcular densidad (D) usando k-ésimo vecino más cercano
        k = min(self.k, size - 1)  # Ajustar k si es mayor que el tamaño - 1

        # Matriz de distancias entre todos los individuos
        distances = np.array([
            [float('inf') if i == j else self._calculate_distance(individuals[i], individuals[j])
             for j in range(size)]
            for i in range(size)
        ])

        # Calcular densidad
        D = np.zeros(size)
        for i in range(size):
            # Ordenar distancias y tomar la k-ésima
            sorted_distances = np.sort(distances[i])
            sigma_k = sorted_distances[k] if k < len(sorted_distances) else sorted_distances[-1]
            D[i] = 1.0 / (sigma_k + 2.0)  # +2 para evitar división por cero

        # Asignar fitness final
        for i in range(size):
            individuals[i].fitness = R[i] + D[i]

    def _calculate_distance(self, ind1, ind2):
        """Calcula la distancia euclidiana en el espacio de objetivos."""
        if not hasattr(ind1, 'objective_values') or not hasattr(ind2, 'objective_values'):
            return float('inf')
        return distance.euclidean(ind1.objective_values, ind2.objective_values)

    def _dominates(self, ind1, ind2):
        """
        Determina si ind1 domina a ind2 (minimización).
        Para maximización, invertir los signos de comparación.
        """
        if not hasattr(ind1, 'objective_values') or not hasattr(ind2, 'objective_values'):
            return False

        # Verificar si ind1 es al menos igual en todos los objetivos
        at_least_equal = all(x <= y for x, y in zip(ind1.objective_values, ind2.objective_values))
        # Y estrictamente mejor en al menos un objetivo
        strictly_better = any(x < y for x, y in zip(ind1.objective_values, ind2.objective_values))

        return at_least_equal and strictly_better

    def _select_archive(self, individuals):
        """
        Selecciona individuos para el archivo:
        1. Selecciona todos los no dominados
        2. Si hay más que archive_size, usa truncamiento
        3. Si hay menos, añade individuos dominados con mejor fitness
        """
        # Filtrar por no dominancia
        non_dominated = []
        for i, ind in enumerate(individuals):
            is_dominated = False
            for other in individuals:
                if other != ind and self._dominates(other, ind):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated.append(ind)

        # Ajustar tamaño del archivo
        if len(non_dominated) <= self.archive_size:
            # Si hay espacio disponible, añadir los mejores individuos dominados
            if len(non_dominated) < self.archive_size:
                dominated = sorted([ind for ind in individuals if ind not in non_dominated],
                                   key=lambda x: x.fitness)
                non_dominated.extend(dominated[:self.archive_size - len(non_dominated)])
            return non_dominated
        else:
            # Truncar archivo usando el método de distancia
            return self._truncate_archive(non_dominated)

    def _truncate_archive(self, archive):
        """
        Trunca el archivo eliminando individuos en áreas densas.
        Usa el concepto de distancia al vecino más cercano.
        """
        while len(archive) > self.archive_size:
            size = len(archive)

            # Calcular todas las distancias
            distances = np.full((size, size), float('inf'))
            for i in range(size):
                for j in range(size):
                    if i != j:
                        distances[i][j] = self._calculate_distance(archive[i], archive[j])

            # Para cada individuo, encontrar la distancia al vecino más cercano
            min_distances = np.min(distances, axis=1)

            # Para individuos con la misma distancia mínima, calcular la segunda más cercana
            to_remove = -1
            current_min = float('inf')

            for i in range(size):
                if min_distances[i] < current_min:
                    to_remove = i
                    current_min = min_distances[i]
                elif min_distances[i] == current_min:
                    # Desempate con la segunda distancia más cercana
                    i_distances = sorted(distances[i])
                    j_distances = sorted(distances[to_remove])

                    k = 1  # Empezamos con la segunda más cercana
                    while k < len(i_distances) and k < len(j_distances):
                        if i_distances[k] < j_distances[k]:
                            to_remove = i
                            break
                        elif i_distances[k] > j_distances[k]:
                            break
                        k += 1

            # Eliminar el individuo con la menor distancia
            if to_remove >= 0:
                archive.pop(to_remove)
            else:
                # Caso extremo: todos tienen la misma distancia, eliminar uno aleatorio
                archive.pop(self._random.randrange(len(archive)))

        return archive




