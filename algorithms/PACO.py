import numpy as np
from copy import deepcopy
from typing import List, Tuple
import json
import random
from collections import defaultdict
from Loaders.LoadStreamingPlans import load_streaming_plan_json
from Loaders.LoadPlatforms import load_platforms_json
from Loaders.LoadUsers import load_users_from_json
from utils.evaluation import calcular_minutos_ponderados, calcular_costo_total
from utils.evaluation import fitness_paco

streamingPlans = load_streaming_plan_json("../Data/streamingPlans.json")
#users = load_users_from_json("../Data/users1.json")

# Cargar películas por plataforma
with open("../Data/MoviesPlatform.json", "r", encoding="utf-8") as f:
    movies_by_platform = json.load(f)

with open("../Data/SeriesPlatform.json", "r", encoding="utf-8") as f:
    series_by_platform = json.load(f)

with open("../Data/indice_plataformas.json", "r", encoding="utf-8") as f:
    platforms_indexed = json.load(f)


class PACOStreaming:
    def __init__(self, n_ants: int, n_iterations: int, n_months: int, n_users: int, users, platform_options: List[int],
                 rho: float = 0.1, alpha: float = 1, beta: float = 3, archive_size: int = 100,
                 no_improvement_generations: int = 10):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.n_months = n_months
        self.n_users = n_users
        self.users = users
        self.platform_options = platform_options
        self.n_platforms = len(platform_options)

        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.archive_size = archive_size

        # Parámetros para terminación por falta de mejora
        self.no_improvement_generations = no_improvement_generations
        self.generations_without_improvement = 0
        self.best_archive_size = 0
        self.best_hypervolume = 0.0
        self.terminated_early = False

        # Feromonas: una matriz de (n_months x n_users x n_platforms)
        self.pheromone = np.ones((self.n_months, self.n_users, self.n_platforms))

        # Archivo de soluciones no dominadas
        self.archive = []

        # Historial para análisis opcional
        self.archive_history = []
        self.trails_history = []

    def optimize(self, fitness_function, args=None):
        self._initialize()

        for iteration in range(self.n_iterations):
            trails = []
            all_solutions = []
            archive_improved = False

            for _ in range(self.n_ants):
                solution = self._construct_solution()
                all_solutions.append(solution.copy())

                # Llamar función de fitness con argumentos si se proporcionan
                if args is not None:
                    objectives = fitness_function(solution, args)
                else:
                    objectives = fitness_function(solution)

                if self._is_pareto_optimal(objectives):
                    old_archive_size = len(self.archive)
                    self._update_archive(solution, objectives)
                    # Si el archivo creció, hubo mejora
                    if len(self.archive) > old_archive_size:
                        archive_improved = True

                trails.append((solution, objectives))

            self._update_pheromone(trails)
            self.archive_history.append(deepcopy(self.archive))
            self.trails_history.append(deepcopy(trails))

            # Verificar si hubo mejora en esta iteración
            current_quality = self._calculate_archive_quality()
            improvement_detected = self._check_improvement(current_quality, archive_improved)

            if improvement_detected:
                self.generations_without_improvement = 0
            else:
                self.generations_without_improvement += 1


            print(f"Iteración {iteration + 1}/{self.n_iterations}, "
                  f"Tamaño del archivo: {len(self.archive)}, "
                  f"Generaciones sin mejora: {self.generations_without_improvement}")

            # Verificar criterio de terminación
            if self.generations_without_improvement >= self.no_improvement_generations:
                print(f"\nTerminación temprana en iteración {iteration + 1}")
                print(f"No se detectaron mejoras durante {self.no_improvement_generations} generaciones consecutivas")
                self.terminated_early = True
                break

        self.all_solutions = all_solutions
        return self.archive

    def _initialize(self):
        self.archive = []
        self.archive_history = []
        self.trails_history = []
        self.generations_without_improvement = 0
        self.best_archive_size = 0
        self.best_hypervolume = 0.0
        self.terminated_early = False

    def _calculate_archive_quality(self) -> float:
        """
        Calcula una métrica de calidad del archivo actual.
        Combina el tamaño del archivo con el hipervolumen aproximado.
        """
        if not self.archive:
            return 0.0

        # Componente 1: Tamaño del archivo (diversidad)
        size_component = len(self.archive)

        # Componente 2: Hipervolumen aproximado (calidad)
        hypervolume = self._approximate_hypervolume()

        # Combinar ambos componentes
        quality = 0.3 * size_component + 0.7 * hypervolume

        return quality

    def _approximate_hypervolume(self) -> float:
        """
        Calcula una aproximación del hipervolumen del frente de Pareto.
        """
        if not self.archive:
            return 0.0

        # Extraer objetivos
        objectives = np.array([obj for _, obj in self.archive])

        if len(objectives) == 1:
            return objectives[0][0] / max(1, objectives[0][1])  # minutos/costo

        # Puntos de referencia para hipervolumen
        # Para minutos (maximizar): usar el mínimo como referencia
        # Para costo (minimizar): usar el máximo como referencia
        ref_minutos = np.min(objectives[:, 0]) - 1
        ref_costo = np.max(objectives[:, 1]) + 1

        # Calcular hipervolumen aproximado usando el método de los rectángulos
        # Ordenar por minutos descendente
        sorted_indices = np.argsort(-objectives[:, 0])
        sorted_objectives = objectives[sorted_indices]

        hypervolume = 0.0
        prev_costo = ref_costo

        for minutos, costo in sorted_objectives:
            if costo < prev_costo:
                # Área del rectángulo
                width = minutos - ref_minutos
                height = prev_costo - costo
                hypervolume += width * height
                prev_costo = costo

        return max(0.0, hypervolume)

    def _check_improvement(self, current_quality: float, archive_grew: bool) -> bool:
        """
        Determina si hubo mejora significativa en esta iteración.
        """
        # Mejora detectada si:
        # 1. El archivo creció (nuevas soluciones no dominadas)
        # 2. La calidad general mejoró significativamente

        if archive_grew:
            return True

        # Umbral de mejora mínima (1% de mejora en calidad)
        improvement_threshold = 0.01

        if current_quality > self.best_hypervolume * (1 + improvement_threshold):
            self.best_hypervolume = current_quality
            return True

        return False

    def _construct_solution(self) -> np.ndarray:
        solution = np.zeros(self.n_months * self.n_users, dtype=int)
        epsilon = 0.15  # 15% de exploración aleatoria

        for month in range(self.n_months):
            for user in range(self.n_users):
                if np.random.rand() < epsilon:
                    # Exploración aleatoria pura
                    platform_choice = np.random.choice(self.platform_options)
                else:
                    # Heurística basada en solución parcial
                    heuristic_info = self._calculate_heuristic_info(month, user, solution)

                    # Evitar valores de feromona cero para prevenir divisiones por cero
                    pheromone_values = np.maximum(self.pheromone[month, user], 1e-10)
                    heuristic_values = np.maximum(heuristic_info, 1e-10)

                    combined_info = (pheromone_values ** self.alpha) * (heuristic_values ** self.beta)

                    # Normalizar probabilidades
                    total = np.sum(combined_info)
                    if total > 0:
                        probs = combined_info / total
                    else:
                        probs = np.ones(self.n_platforms) / self.n_platforms

                    # Selección basada en probabilidad
                    try:
                        platform_idx = np.random.choice(self.n_platforms, p=probs)
                        platform_choice = self.platform_options[platform_idx]
                    except ValueError:
                        # Fallback en caso de problemas con las probabilidades
                        platform_choice = np.random.choice(self.platform_options)

                solution[month * self.n_users + user] = platform_choice

        return solution

    def _calculate_heuristic_info(self, current_month, current_user, partial_solution) -> np.ndarray:
        """Calcula información heurística para un usuario y mes específicos"""
        heuristic = np.ones(self.n_platforms)

        user = self.users[current_user]

        for plat_idx, platform_id in enumerate(self.platform_options):
            platform_name = platforms_indexed.get(str(platform_id), "")

            # Si es 0 (ninguna plataforma), valor base bajo
            if platform_id == 0:
                content_price_ratio = 0.2
            else:
                platform_data = streamingPlans.get(str(platform_id), None)
                if platform_data:
                    if isinstance(platform_data, list):
                        price = min(p["precio"] for p in platform_data if "precio" in p)
                    else:
                        price = platform_data.get("precio", 10000)

                    potential_content = self._calculate_potential_content(user, platform_name)
                    content_price_ratio = potential_content / max(1, price)
                else:
                    content_price_ratio = 0.1

            # Factor de continuidad mejorado
            continuity_factor = 1.0
            if current_month > 0:
                prev_month_idx = (current_month - 1) * self.n_users + current_user
                if prev_month_idx < len(partial_solution):
                    prev_platform = partial_solution[prev_month_idx]
                    if prev_platform == platform_id and platform_id != 0:
                        continuity_factor = 1.15  # Pequeño bonus por continuidad

            # Factor de diversidad: penalizar si ya se usa mucho esta plataforma
            diversity_factor = 1.0
            if current_month > 2:  # Solo después de algunos meses
                usage_count = 0
                for past_month in range(max(0, current_month - 3), current_month):
                    past_idx = past_month * self.n_users + current_user
                    if past_idx < len(partial_solution) and partial_solution[past_idx] == platform_id:
                        usage_count += 1

                if usage_count >= 2:  # Si se ha usado 2+ veces en los últimos 3 meses
                    diversity_factor = 0.9  # Pequeña penalización

            heuristic[plat_idx] = content_price_ratio * continuity_factor * diversity_factor

        return heuristic

    def _calculate_potential_content(self, user, platform_name):
        """Calcula el contenido potencial de interés de un usuario para una plataforma"""
        potential_content = 0

        # Procesar películas
        for movie in user.movies:
            movie_interest = movie.get("interest", 1.0)
            movie_platforms = movie.get("platforms", [])

            if platform_name in movie_platforms:
                potential_content += movie_interest * 10

        # Procesar series
        for serie in user.series:
            serie_interest = serie.get("interest", 1.0)
            serie_platforms = serie.get("platforms", [])

            if platform_name in serie_platforms:
                potential_content += serie_interest * 15

            # Revisar temporadas
            for temporada in serie.get("season", []):
                temporada_platforms = temporada.get("platforms", [])
                if platform_name in temporada_platforms:
                    potential_content += serie_interest * 8

        return max(0.1, potential_content)

    def _is_pareto_optimal(self, new_objectives: List[float]) -> bool:
        """Determina si una solución es Pareto óptima respecto al archivo existente"""
        for _, objectives in self.archive:
            if self._dominates(objectives, new_objectives):
                return False
        return True

    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """
        Determina si obj1 domina a obj2:
        - Objetivo 0 (minutos ponderados): MAXIMIZAR
        - Objetivo 1 (costo total): MINIMIZAR
        """
        # Validación de entrada
        if len(obj1) != 2 or len(obj2) != 2:
            return False

        minutos_mejor_o_igual = obj1[0] >= obj2[0]
        costo_mejor_o_igual = obj1[1] <= obj2[1]
        mejor_o_igual = minutos_mejor_o_igual and costo_mejor_o_igual

        minutos_estrictamente_mejor = obj1[0] > obj2[0]
        costo_estrictamente_mejor = obj1[1] < obj2[1]
        estrictamente_mejor = minutos_estrictamente_mejor or costo_estrictamente_mejor

        return mejor_o_igual and estrictamente_mejor

    def _update_archive(self, solution: np.ndarray, objectives: List[float]):
        # Eliminar soluciones dominadas por la nueva
        self.archive = [(sol, obj) for sol, obj in self.archive if not self._dominates(objectives, obj)]

        # Añadir la nueva solución no dominada
        self.archive.append((solution.copy(), objectives))

        # Truncar el archivo si excede el tamaño máximo
        if len(self.archive) > self.archive_size:
            self._truncate_archive()

    def _truncate_archive(self):
        """Trunca el archivo basado en distancias de crowding"""
        if len(self.archive) <= self.archive_size:
            return

        distances = self._calculate_crowding_distance()
        if len(distances) == 0:
            return

        sorted_indices = np.argsort(-distances)
        self.archive = [self.archive[i] for i in sorted_indices[:self.archive_size]]

    def _calculate_crowding_distance(self) -> np.ndarray:
        """Calcula las distancias de crowding para mantener diversidad"""
        n_solutions = len(self.archive)

        if n_solutions <= 2:
            return np.full(n_solutions, np.inf)

        distances = np.zeros(n_solutions)
        fitnesses = np.array([obj for _, obj in self.archive])

        for m in range(2):  # 2 objetivos
            indices = np.argsort(fitnesses[:, m])

            # Extremos tienen distancia infinita
            distances[indices[0]] = distances[indices[-1]] = np.inf

            # Rango del objetivo para normalización
            obj_range = fitnesses[indices[-1], m] - fitnesses[indices[0], m]

            if obj_range > 1e-10:  # Evitar división por cero
                for i in range(1, n_solutions - 1):
                    distances[indices[i]] += (
                                                     fitnesses[indices[i + 1], m] - fitnesses[indices[i - 1], m]
                                             ) / obj_range

        return distances

    def _update_pheromone(self, trails: List[Tuple[np.ndarray, List[float]]]):
        """Actualiza las feromonas con estrategia mejorada"""
        if not trails:
            return

        # Evaporación global
        self.pheromone *= (1 - self.rho)

        # Extraer objetivos
        minutos_values = np.array([obj[0] for _, obj in trails])
        costo_values = np.array([obj[1] for _, obj in trails])

        # Evitar problemas de normalización
        minutos_range = max(1e-10, np.max(minutos_values) - np.min(minutos_values))
        costo_range = max(1e-10, np.max(costo_values) - np.min(costo_values))

        # Normalizar objetivos
        normalized_scores = []
        for _, obj in trails:
            if minutos_range > 1e-10:
                norm_minutos = (obj[0] - np.min(minutos_values)) / minutos_range
            else:
                norm_minutos = 0.5

            if costo_range > 1e-10:
                norm_costo = 1 - ((obj[1] - np.min(costo_values)) / costo_range)
            else:
                norm_costo = 0.5

            # Score combinado ponderado
            combined_score = 0.6 * norm_minutos + 0.4 * norm_costo
            normalized_scores.append(combined_score)

        # Actualizar feromonas solo para las mejores soluciones
        top_k = min(max(1, self.n_ants // 3), len(trails))  # Top 33% de hormigas
        ranked_indices = np.argsort(normalized_scores)[-top_k:]

        for rank, idx in enumerate(ranked_indices):
            solution, _ = trails[idx]
            reinforcement = (rank + 1) / top_k  # Refuerzo proporcional al ranking

            for month in range(self.n_months):
                for user in range(self.n_users):
                    sol_idx = month * self.n_users + user
                    if sol_idx < len(solution):
                        platform = solution[sol_idx]
                        try:
                            platform_idx = self.platform_options.index(platform)
                            # Actualización con límite superior para evitar valores extremos
                            self.pheromone[month, user, platform_idx] += min(reinforcement, 2.0)
                            # Mantener un mínimo de feromona
                            self.pheromone[month, user, platform_idx] = max(
                                self.pheromone[month, user, platform_idx], 0.01
                            )
                        except ValueError:
                            # Platform no está en platform_options
                            continue


def fitness_paco(solution, args):
    """Función de fitness mejorada con manejo de errores"""
    try:
        n_users = len(args['users'])
        n_months = 12

        candidate = []
        for user_idx in range(n_users):
            user_platforms = []
            for month in range(n_months):
                index = month * n_users + user_idx
                if index < len(solution):
                    user_platforms.append(solution[index])
                else:
                    user_platforms.append(0)  # Default: ninguna plataforma
            candidate.append(user_platforms)

        minutos_ponderados = calcular_minutos_ponderados(candidate, args)
        costo_total = calcular_costo_total(candidate, args)

        return [minutos_ponderados, costo_total]

    except Exception as e:
        print(f"Error en fitness_paco: {e}")
        return [0.0, float('inf')]  # Solución inválida