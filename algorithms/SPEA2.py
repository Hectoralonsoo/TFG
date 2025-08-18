from inspyred.ec import EvolutionaryComputation
from inspyred.ec import Individual
import numpy as np
from scipy.spatial import distance
from inspyred.ec import selectors, variators, terminators
from utils.logging_custom import get_non_dominated


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
        self.generations_without_improvement = 0
        self.previous_pareto_count = 0  # Contador de soluciones no dominadas
        self.previous_solutions = set()  # Conjunto de soluciones de la generación anterior

    def evolve(self, pop_size, maximize, max_generations, num_selected, **kwargs):
        """
        Implementación de SPEA2 que termina SOLO cuando no hay mejora por 10 generaciones.
        """
        self._maximize = maximize
        self.num_generations = 0
        self._num_evaluations = 0
        self.archive = []
        kwargs['max_generations'] = max_generations

        max_generations_without_improvement = 10
        self.generations_without_improvement = 0
        self.previous_pareto_count = 0
        self.previous_solutions = set()

        generation = 0
        population = [Individual(self.generator(random=self._random, args=kwargs)) for _ in range(pop_size)]

        kwargs['_individuals'] = population
        population = self._evaluate_population(population, kwargs)

        # Calcular k dinámicamente para densidad
        self.k = kwargs.get('k', max(1, int(len(population) ** 0.5)))

        print(f"=== INICIANDO SPEA2 ===")
        print(f"Criterio de terminación: {max_generations_without_improvement} generaciones sin mejora")
        print(f"Máximo de generaciones (límite superior): {max_generations}")
        print(f"======================")

        # Variable para saber si terminó por no mejora
        terminated_by_no_improvement = False

        while generation < max_generations:
            print(f"\n--- Generación {generation} ---")
            print(
                f"🔄 Generaciones sin mejora: {self.generations_without_improvement}/{max_generations_without_improvement}")

            # Combinar población actual con archivo
            combined = population + self.archive

            # Asignar fitness SPEA2 a todos los individuos
            self._assign_spea2_fitness(combined)

            # Seleccionar nuevo archivo (frente de Pareto)
            self.archive = self._select_archive(combined)

            # DEBUG: Verificar que tenemos archivo
            if not self.archive:
                print("WARNING: Archivo vacío en generación", generation)
                # Crear archivo con los mejores individuos si está vacío
                if combined:
                    sorted_combined = sorted(combined, key=lambda x: x.fitness)
                    self.archive = sorted_combined[:min(10, len(sorted_combined))]
                    print(f"Archivo de emergencia creado con {len(self.archive)} individuos")

            # Verificar terminación por no mejora
            should_terminate = self._check_improvement_termination(max_generations_without_improvement, kwargs)

            if self.observer:
                self.observer(self.archive, generation, self._num_evaluations, kwargs)

            # *** AQUÍ ESTÁ EL PUNTO CLAVE ***
            # Terminar SOLO si no hay mejora por 10 generaciones
            if should_terminate:
                print(f"\n🚨 TERMINANDO en generación {generation}")
                print(f"🚨 MOTIVO: Sin mejora por {max_generations_without_improvement} generaciones")
                print(f"🚨 Generaciones sin mejora: {self.generations_without_improvement}")
                terminated_by_no_improvement = True
                break

            # Mostrar estado actual después de verificar mejora
            print(
                f"📊 Estado actual: {self.generations_without_improvement}/{max_generations_without_improvement} generaciones sin mejora")

            # Selección de padres del archivo
            mating_pool = self._select_parents(self.archive, num_selected)

            if not mating_pool:
                print("WARNING: No se pudieron seleccionar padres")
                break

            offspring = []

            # Generar descendencia
            while len(offspring) < pop_size:
                if len(mating_pool) >= 2:
                    selected = self._random.sample(mating_pool, 2)
                elif len(mating_pool) == 1:
                    selected = [mating_pool[0], mating_pool[0]]
                else:
                    break

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

        print(f"\n=== ALGORITMO TERMINADO ===")
        print(f"Generación final: {generation}")
        print(f"Generaciones sin mejora final: {self.generations_without_improvement}")
        if terminated_by_no_improvement:
            print(f"✅ TERMINÓ POR: No mejora por {max_generations_without_improvement} generaciones")
        else:
            print(f"❌ TERMINÓ POR: Límite máximo de generaciones ({max_generations})")
        print(f"Tamaño final del archivo: {len(self.archive)}")
        print(f"Evaluaciones totales: {self._num_evaluations}")
        print(f"==========================")

        # Evaluación final del archivo
        if self.archive:
            kwargs['_individuals'] = self.archive
            self._evaluate_population(self.archive, kwargs)

        return self.archive

    def _check_improvement_termination(self, max_generations_without_improvement, kwargs):
        """
        Verifica si debe terminar por no mejora - VERSION SIMPLIFICADA
        Detecta mejora si hay cambios en el frente de Pareto (nuevas soluciones o dominancia)
        """
        if not self.archive:
            print("  ⚠️  Archivo vacío, no se puede evaluar mejora")
            return False

        # Obtener soluciones actuales del archivo
        current_solutions = set()
        for ind in self.archive:
            if hasattr(ind, 'objective_values') and ind.objective_values:
                current_solutions.add(tuple(ind.objective_values))

        current_pareto_count = len(current_solutions)
        print(f"  📊 Soluciones no dominadas: {current_pareto_count}")

        # Si es la primera generación, inicializar
        if not hasattr(self, 'previous_solutions'):
            self.previous_solutions = current_solutions.copy()
            self.previous_pareto_count = current_pareto_count
            self.generations_without_improvement = 0
            print(f"  🔄 Primera generación, inicializando con {current_pareto_count} soluciones")
            return False

        # Verificar si hay cambios en el frente de Pareto
        # Nuevas soluciones que no estaban antes
        new_solutions = current_solutions - self.previous_solutions
        # Soluciones que desaparecieron (fueron dominadas)
        removed_solutions = self.previous_solutions - current_solutions

        # Hay mejora si:
        # 1. Se encontraron nuevas soluciones, O
        # 2. Se eliminaron soluciones (porque fueron dominadas por mejores)
        has_improvement = len(new_solutions) > 0 or len(removed_solutions) > 0

        if has_improvement:
            self.generations_without_improvement = 0

            improvement_details = []
            if len(new_solutions) > 0:
                improvement_details.append(f"{len(new_solutions)} nueva(s) solución(es)")
            if len(removed_solutions) > 0:
                improvement_details.append(f"{len(removed_solutions)} solución(es) dominada(s)")

            print(f"  ✅ MEJORA detectada: {', '.join(improvement_details)}")
            print(f"  ✅ Soluciones: {self.previous_pareto_count} → {current_pareto_count}")
            print(f"  ✅ Contador de generaciones sin mejora: {self.generations_without_improvement} (RESET)")
        else:
            self.generations_without_improvement += 1
            print(f"  ❌ Sin mejora detectada - Frente de Pareto sin cambios")
            print(
                f"  ❌ Contador de generaciones sin mejora: {self.generations_without_improvement}/{max_generations_without_improvement}")
            print(f"     Soluciones: {current_pareto_count} (mismo conjunto que generación anterior)")

        # Actualizar para la próxima generación
        self.previous_solutions = current_solutions.copy()
        self.previous_pareto_count = current_pareto_count

        # Verificar si debe terminar
        should_terminate = self.generations_without_improvement >= max_generations_without_improvement

        if should_terminate:
            print(f"  🚨 CRITERIO DE TERMINACIÓN ALCANZADO!")
            print(f"  🚨 {self.generations_without_improvement} >= {max_generations_without_improvement}")

        return should_terminate

    def _calculate_hypervolume_approximation(self):
        """
        Calcula una aproximación del hipervolumen más robusta.
        (Mantenido por compatibilidad, pero no se usa en el criterio de terminación)
        """
        if not self.archive:
            return 0.0

        try:
            # Aproximación mejorada del hipervolumen
            total = 0.0
            for ind in self.archive:
                if hasattr(ind, 'objective_values') and ind.objective_values:
                    # Sumar el inverso de la norma euclidiana (para minimización)
                    norm = np.linalg.norm(ind.objective_values)
                    if norm > 0:
                        total += 1.0 / (norm + 1e-10)  # Evitar división por cero

            return total
        except Exception as e:
            print(f"  ⚠️  Error calculando hipervolumen: {e}")
            return 0.0

    def _evaluate_population(self, population, kwargs):
        """Evalúa la población y asigna valores objetivo."""
        if population:
            candidates = [p.candidate for p in population]
            fitness_values = self.evaluator(candidates=candidates, args=kwargs)

            for index, individual in enumerate(population):
                individual.objective_values = fitness_values[index]

        return population

    def _select_parents(self, archive, num_selected):
        """Selección de padres mejorada para manejar archivos pequeños."""
        selected = []
        if len(archive) == 0:
            return selected

        for _ in range(num_selected):
            if len(archive) == 1:
                selected.append(archive[0])
            elif len(archive) >= 2:
                competitors = self._random.sample(archive, 2)
                winner = min(competitors, key=lambda x: getattr(x, 'fitness', float('inf')))
                selected.append(winner)
            else:
                # Fallback
                selected.append(self._random.choice(archive))

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
            R[i] = sum(S[j] for j in range(size) if dominance[j][i])

        # Calcular densidad (D) usando k-ésimo vecino más cercano
        k = min(self.k, size - 1)
        if k <= 0:
            k = 1

        # Matriz de distancias entre todos los individuos
        distances = np.array([
            [float('inf') if i == j else self._calculate_distance(individuals[i], individuals[j])
             for j in range(size)]
            for i in range(size)
        ])

        # Calcular densidad
        D = np.zeros(size)
        for i in range(size):
            sorted_distances = np.sort(distances[i])
            if len(sorted_distances) > k:
                sigma_k = sorted_distances[k]
            else:
                sigma_k = sorted_distances[-1] if len(sorted_distances) > 0 else 1.0
            D[i] = 1.0 / (sigma_k + 2.0)

        # Asignar fitness final
        for i in range(size):
            individuals[i].fitness = R[i] + D[i]

    def _calculate_distance(self, ind1, ind2):
        """Calcula la distancia euclidiana en el espacio de objetivos."""
        if not hasattr(ind1, 'objective_values') or not hasattr(ind2, 'objective_values'):
            return float('inf')
        try:
            return distance.euclidean(ind1.objective_values, ind2.objective_values)
        except:
            return float('inf')

    def _dominates(self, ind1, ind2):
        """Determina si ind1 domina a ind2."""
        if not hasattr(ind1, 'objective_values') or not hasattr(ind2, 'objective_values'):
            return False

        if len(ind1.objective_values) != len(ind2.objective_values):
            return False

        try:
            return (all(x <= y for x, y in zip(ind1.objective_values, ind2.objective_values)) and
                    any(x < y for x, y in zip(ind1.objective_values, ind2.objective_values)))
        except:
            return False

    def _get_pareto_front(self, individuals):
        """Identifica el frente de Pareto."""
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
        """Selecciona SOLO individuos no dominados para el archivo."""
        if not individuals:
            return []

        non_dominated = self._get_pareto_front(individuals)
        non_dominated = self._deduplicate_objectives(non_dominated)
        self._copy_attributes(individuals, non_dominated)

        print(f"  📋 Frente de Pareto: {len(non_dominated)} individuos")

        # Si hay demasiados individuos no dominados, truncar por densidad
        if len(non_dominated) > self.archive_size:
            print(f"  ✂️  Truncando archivo de {len(non_dominated)} a {self.archive_size}")
            return self._truncate_archive(non_dominated)

        return non_dominated

    def _deduplicate_objectives(self, individuals):
        """Elimina individuos con objetivos repetidos."""
        seen = set()
        unique = []
        for ind in individuals:
            if hasattr(ind, 'objective_values'):
                key = tuple(ind.objective_values)
                if key not in seen:
                    seen.add(key)
                    unique.append(ind)
        return unique

    def _copy_attributes(self, source_individuals, target_individuals):
        """Copia atributos como watched_movies y watched_series entre individuos."""
        source_dict = {}
        for ind in source_individuals:
            if hasattr(ind, 'candidate'):
                key = self._get_candidate_key(ind.candidate)
                source_dict[key] = ind

        for target_ind in target_individuals:
            if hasattr(target_ind, 'candidate'):
                key = self._get_candidate_key(target_ind.candidate)
                if key in source_dict:
                    source_ind = source_dict[key]
                    if hasattr(source_ind, 'watched_movies'):
                        target_ind.watched_movies = source_ind.watched_movies
                    if hasattr(source_ind, 'watched_series'):
                        target_ind.watched_series = source_ind.watched_series

    def _get_candidate_key(self, candidate):
        """Crea una clave única para un candidato."""
        try:
            if isinstance(candidate, list):
                if any(isinstance(item, list) for item in candidate):
                    return tuple(tuple(item) if isinstance(item, list) else item for item in candidate)
                else:
                    return tuple(candidate)
            else:
                return candidate
        except:
            return str(candidate)

    def _truncate_archive(self, archive):
        """Trunca el archivo eliminando individuos en áreas densas."""
        while len(archive) > self.archive_size:
            size = len(archive)

            distances = np.full((size, size), float('inf'))
            for i in range(size):
                for j in range(size):
                    if i != j:
                        distances[i][j] = self._calculate_distance(archive[i], archive[j])

            min_distances = np.min(distances, axis=1)

            to_remove = -1
            current_min = float('inf')

            for i in range(size):
                if min_distances[i] < current_min:
                    to_remove = i
                    current_min = min_distances[i]
                elif min_distances[i] == current_min:
                    i_distances = sorted(distances[i])
                    j_distances = sorted(distances[to_remove])

                    k = 1
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