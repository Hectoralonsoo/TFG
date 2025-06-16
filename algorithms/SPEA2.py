from inspyred.ec import EvolutionaryComputation
from inspyred.ec import Individual
import numpy as np
from scipy.spatial import distance
from inspyred.ec import selectors, variators, terminators
from utils.logging import get_non_dominated


class SPEA2(EvolutionaryComputation):
    def __init__(self, random):
        super().__init__(random)
        self.archive = []
        self.archive_size = 300
        self.k = 3
        self.default_selector = selectors.tournament_selection
        self.default_variator = [variators.uniform_crossover, variators.random_reset_mutation]
        self.default_terminator = terminators.no_improvement_termination

        # Variables para el terminador de no mejora
        self.best_fitness_history = []
        self.generations_without_improvement = 0

    # Método mejorado para el bucle principal de evolución
    def evolve(self, pop_size, maximize, max_generations, num_selected, **kwargs):
        """
        Implementación corregida del algoritmo SPEA2 con terminador de no mejora.
        """
        self._maximize = maximize
        self.num_generations = 0
        self._num_evaluations = 0
        self.archive = []
        kwargs['max_generations'] = max_generations

        # Parámetros para el terminador de no mejora
        max_generations_without_improvement = kwargs.get('max_generations_without_improvement', 10)
        self.best_fitness_history = []
        self.generations_without_improvement = 0

        generation = 0
        population = [Individual(self.generator(random=self._random, args=kwargs)) for _ in range(pop_size)]

        kwargs['_individuals'] = population
        population = self._evaluate_population(population, kwargs)

        # Calcular k dinámicamente para densidad
        self.k = kwargs.get('k', max(1, int(len(population) ** 0.5)))

        while generation < max_generations:
            # Combinar población actual con archivo
            combined = population + self.archive

            # Asignar fitness SPEA2 a todos los individuos
            self._assign_spea2_fitness(combined)

            # Seleccionar nuevo archivo (debería contener el frente de Pareto)
            self.archive = self._select_archive(combined)

            # Verificar terminación por no mejora
            should_terminate = self._check_no_improvement_termination(max_generations_without_improvement, kwargs)

            if self.observer:
                self.observer(self, self.archive, generation, self._num_evaluations, kwargs)

            # Si debe terminar por no mejora, salir del bucle
            if should_terminate:
                print(
                    f"Terminando en generación {generation} por no mejora durante {max_generations_without_improvement} generaciones")
                break

            # Selección de padres del archivo
            mating_pool = self._select_parents(self.archive, num_selected)
            offspring = []

            # Generar descendencia
            while len(offspring) < pop_size:
                selected = self._random.sample(mating_pool, 2)
                parents = [p.candidate for p in selected]
                children = parents

                for v in self.variator:
                    children = v(self._random, children, {'random': self._random, '_ec': self, **kwargs})

                offspring.extend([Individual(child) for child in children])

            offspring = offspring[:pop_size]

            # Evaluar descendencia
            kwargs['_individuals'] = offspring
            offspring = self._evaluate_population(offspring, kwargs)
            population = offspring

            generation += 1
            self.num_generations = generation
            self._num_evaluations += len(population)

        # Evaluación final del archivo
        kwargs['_individuals'] = self.archive
        self._evaluate_population(self.archive, kwargs)

        return self.archive

    def _check_no_improvement_termination(self, max_generations_without_improvement, kwargs):
        """
        Verifica si debe terminar por no mejora en el fitness.
        Para problemas multiobjetivo, usa el hipervolumen o número de soluciones no dominadas.
        """
        if not self.archive:
            return False

        # Para SPEA2, usamos como métrica de mejora el mejor fitness del archivo
        # (menor fitness es mejor en SPEA2)
        current_best_fitness = min(ind.fitness for ind in self.archive)

        # Si es la primera generación, inicializar
        if not self.best_fitness_history:
            self.best_fitness_history.append(current_best_fitness)
            self.generations_without_improvement = 0
            return False

        # Verificar si hay mejora (en SPEA2, menor fitness es mejor)
        previous_best = self.best_fitness_history[-1]

        # Tolerancia para considerar una mejora significativa
        tolerance = kwargs.get('improvement_tolerance', 1e-6)

        if current_best_fitness < (previous_best - tolerance):
            # Hay mejora
            self.generations_without_improvement = 0
            print(f"Mejora detectada: {previous_best:.6f} -> {current_best_fitness:.6f}")
        else:
            # No hay mejora
            self.generations_without_improvement += 1
            print(
                f"Sin mejora por {self.generations_without_improvement} generaciones (best: {current_best_fitness:.6f})")

        # Agregar el fitness actual al historial
        self.best_fitness_history.append(current_best_fitness)

        # Verificar si debe terminar
        return self.generations_without_improvement >= max_generations_without_improvement

    def _evaluate_population(self, population, kwargs):
        """Evalúa la población y asigna valores objetivo."""
        if population:
            candidates = [p.candidate for p in population]
            # Evaluar con el evaluador y pasar la población actual
            fitness_values = self.evaluator(candidates=candidates, args=kwargs)

            for index, individual in enumerate(population):
                individual.objective_values = fitness_values[index]

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
        Determina si ind1 domina a ind2 usando la MISMA lógica que get_non_dominated().
        """
        if not hasattr(ind1, 'objective_values') or not hasattr(ind2, 'objective_values'):
            return False

        if len(ind1.objective_values) != len(ind2.objective_values):
            return False

        # Usar la MISMA lógica que tu función get_non_dominated
        return (all(x <= y for x, y in zip(ind1.objective_values, ind2.objective_values)) and
                any(x < y for x, y in zip(ind1.objective_values, ind2.objective_values)))

    def _get_pareto_front(self, individuals):
        """
        Identifica el frente de Pareto usando la MISMA lógica que get_non_dominated().
        """
        pareto_front = []

        for ind in individuals:
            dominated = False
            for other in individuals:
                if other != ind and self._dominates(other, ind):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(ind)

        return pareto_front

    def _select_archive(self, individuals):
        """
        Selecciona SOLO individuos no dominados para el archivo.
        El archivo contendrá únicamente el frente de Pareto.
        """
        if not individuals:
            return []

        # Encontrar el frente de Pareto (individuos no dominados)
        non_dominated = self._get_pareto_front(individuals)

        non_dominated = self._deduplicate_objectives(non_dominated)

        # Copiar atributos útiles
        self._copy_attributes(individuals, non_dominated)

        print(f"Frente de Pareto encontrado: {len(non_dominated)} individuos")

        # Si hay demasiados individuos no dominados, truncar por densidad
        if len(non_dominated) > self.archive_size:
            print(f"Truncando de {len(non_dominated)} a {self.archive_size} por densidad")
            return self._truncate_archive(non_dominated)

        # Devolver SOLO el frente de Pareto (sin completar con dominados)

        print(f"Archivo final: {len(non_dominated)} individuos (solo no dominados)")
        return non_dominated

    def _deduplicate_objectives(self, individuals):
        """
        Elimina individuos con objetivos repetidos (duplicados en el frente).
        """
        seen = set()
        unique = []
        for ind in individuals:
            key = tuple(ind.objective_values)
            if key not in seen:
                seen.add(key)
                unique.append(ind)
        return unique

    def _select_parents(self, archive, num_selected):
        """
        Selección de padres mejorada para manejar archivos pequeños.
        """
        selected = []
        if len(archive) == 0:
            return selected

        # Si el archivo es muy pequeño, permitir selección con reemplazo
        for _ in range(num_selected):
            if len(archive) == 1:
                # Solo hay un individuo, seleccionarlo siempre
                selected.append(archive[0])
            elif len(archive) >= 2:
                # Torneo binario normal
                competitors = self._random.sample(archive, 2)
                winner = min(competitors, key=lambda x: x.fitness)
                selected.append(winner)

        return selected

    def _debug_get_non_dominated(self, solutions):
        """
        Función de debug que replica exactamente tu get_non_dominated()
        """
        pareto = []
        for ind in solutions:
            dominated = False
            for other in solutions:
                if (other != ind and
                        all(x <= y for x, y in zip(other.objective_values, ind.objective_values)) and
                        any(x < y for x, y in zip(other.objective_values, ind.objective_values))):
                    dominated = True
                    break
            if not dominated:
                pareto.append(ind)
        return pareto

    # Método adicional para debug completo
    def debug_archive_consistency(self):
        """
        Verifica la consistencia entre el archivo y get_non_dominated()
        """
        if not self.archive:
            return

        # Aplicar get_non_dominated al archivo actual
        external_pareto = self._debug_get_non_dominated(self.archive)

        print(f"Tamaño del archivo: {len(self.archive)}")
        print(f"Frente de Pareto externo: {len(external_pareto)}")

        # Verificar si todos los del archivo están en el frente externo
        archive_in_pareto = sum(1 for ind in self.archive if ind in external_pareto)
        print(f"Individuos del archivo que están en frente externo: {archive_in_pareto}")

        if len(external_pareto) != archive_in_pareto:
            print("WARNING: El archivo contiene individuos dominados que no deberían estar")

            # Mostrar los que no están
            not_in_pareto = [ind for ind in self.archive if ind not in external_pareto]
            print(f"Individuos dominados en archivo: {len(not_in_pareto)}")
            for i, ind in enumerate(not_in_pareto[:3]):  # Mostrar solo los primeros 3
                print(f"  Dominado {i}: objectives={ind.objective_values}, fitness={ind.fitness}")

    def _copy_attributes(self, source_individuals, target_individuals):
        """
        Copia atributos como watched_movies y watched_series entre individuos
        """
        # Crear un diccionario para búsqueda rápida por candidate
        source_dict = {}
        for ind in source_individuals:
            # Usar una tupla de los valores del candidato como clave
            if hasattr(ind, 'candidate'):
                key = self._get_candidate_key(ind.candidate)
                source_dict[key] = ind

        # Copiar atributos a los individuos target
        for target_ind in target_individuals:
            key = self._get_candidate_key(target_ind.candidate)
            if key in source_dict:
                source_ind = source_dict[key]
                # Copiar watched_movies si existe
                if hasattr(source_ind, 'watched_movies'):
                    target_ind.watched_movies = source_ind.watched_movies
                # Copiar watched_series si existe
                if hasattr(source_ind, 'watched_series'):
                    target_ind.watched_series = source_ind.watched_series

    def _get_candidate_key(self, candidate):
        """
        Crea una clave única para un candidato
        """
        # Convertir el candidato a una tupla para usarlo como clave
        if isinstance(candidate, list):
            # Si es una lista anidada, convertir cada elemento interno
            if any(isinstance(item, list) for item in candidate):
                return tuple(tuple(item) if isinstance(item, list) else item for item in candidate)
            else:
                return tuple(candidate)
        else:
            return candidate

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

            if to_remove >= 0:
                archive.pop(to_remove)
            else:
                archive.pop(self._random.randrange(len(archive)))

        return archive