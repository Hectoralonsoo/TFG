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
users = load_users_from_json("../Data/users.json")

# Cargar películas por plataforma
with open("../Data/MoviesPlatform.json", "r") as f:
    movies_by_platform = json.load(f)

with open("../Data/SeriesPlatform.json", "r", encoding="utf-8") as f:
    series_by_platform = json.load(f)

with open("../Data/indice_plataformas.json", "r", encoding="utf-8") as f:
    platforms_indexed = json.load(f)


class PACOStreaming:
    def __init__(self, n_ants: int, n_iterations: int, n_months: int, n_users: int, platform_options: List[int],
                 rho: float = 0.1, alpha: float = 1, beta: float = 3, archive_size: int = 100):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.n_months = n_months
        self.n_users = n_users
        self.platform_options = platform_options
        self.n_platforms = len(platform_options)

        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.archive_size = archive_size

        # Feromonas: una matriz de (n_months x n_users x n_platforms)
        self.pheromone = np.ones((self.n_months, self.n_users, self.n_platforms))

        # Archivo de soluciones no dominadas
        self.archive = []

        # Historial para análisis opcional
        self.archive_history = []
        self.trails_history = []

    def optimize(self, fitness_function):
        self._initialize()

        for iteration in range(self.n_iterations):
            trails = []
            all_solutions = []

            for _ in range(self.n_ants):
                solution = self._construct_solution()
                all_solutions.append(solution.copy())
                objectives = fitness_function(solution)

                if self._is_pareto_optimal(objectives):
                    self._update_archive(solution, objectives)

                trails.append((solution, objectives))

            self._update_pheromone(trails)
            self.archive_history.append(deepcopy(self.archive))
            self.trails_history.append(deepcopy(trails))

            if (iteration + 1) % 10 == 0:
                print(f"Iteración {iteration + 1}/{self.n_iterations}, Tamaño del archivo: {len(self.archive)}")

        self.all_solutions = all_solutions
        return self.archive

    def _initialize(self):
        self.archive = []
        self.archive_history = []
        self.trails_history = []

    def _construct_solution(self) -> np.ndarray:
        solution = np.zeros(self.n_months * self.n_users, dtype=int)
        epsilon = 0.1  # 10% de exploración aleatoria

        for month in range(self.n_months):
            for user in range(self.n_users):
                if np.random.rand() < epsilon:
                    # Exploración aleatoria pura
                    platform_choice = np.random.choice(self.platform_options)
                else:
                    # Heurística basada en solución parcial
                    heuristic_info = self._calculate_heuristic_info(month, user, solution)

                    # BONUS por cambio respecto al mes anterior (incentiva variabilidad)
                    if month > 0:
                        prev_platform = solution[(month - 1) * self.n_users + user]
                        for idx_p, plat_id in enumerate(self.platform_options):
                            if plat_id != prev_platform and plat_id != 0:
                                heuristic_info[idx_p] *= 1.05  # Pequeño incentivo al cambio

                    combined_info = (self.pheromone[month, user] ** self.alpha) * (heuristic_info ** self.beta)
                    probs = combined_info / np.sum(combined_info)

                    platform_choice = np.random.choice(self.platform_options, p=probs)

                solution[month * self.n_users + user] = platform_choice

        return solution

    def _calculate_heuristic_info(self, current_month, current_user, partial_solution) -> np.ndarray:
        """Calcula información heurística para un usuario y mes específicos"""
        # Inicializar vector de información heurística para cada plataforma
        heuristic = np.ones(self.n_platforms)

        user = users[current_user]  # Obtener el usuario actual

        for plat_idx, platform_id in enumerate(self.platform_options):
            # Factor 1: Relación contenido preferido/precio
            platform_name = platforms_indexed.get(str(platform_id), "")

            # Si es 0 (ninguna plataforma), el valor de heurística es bajo pero no cero
            if platform_id == 0:
                content_price_ratio = 0.1
            else:
                # Obtener plan y precio
                platform_data = streamingPlans.get(str(platform_id), None)
                if platform_data:
                    if isinstance(platform_data, list):
                        price = min(p["precio"] for p in platform_data)
                    else:
                        price = platform_data.get("precio", 10000)

                    # Calcular contenido potencial que le interesa al usuario
                    potential_content = self._calculate_potential_content(user, platform_name)

                    # Relación contenido/precio (evitar división por cero)
                    content_price_ratio = potential_content / max(1, price)
                else:
                    content_price_ratio = 0.1

            # Factor 2: Continuidad (mantener la misma plataforma tiene sentido a veces)
            continuity_factor = 1.0
            if current_month > 0:
                prev_month_idx = (current_month - 1) * self.n_users + current_user
                if partial_solution[prev_month_idx] == platform_id and platform_id != 0:
                    continuity_factor = 1.2  # Premio por continuidad

            # Factor 3: Temporalidad (ciertas plataformas pueden ser más atractivas en ciertos meses)
            seasonal_factor = 1.0
            # Ejemplo: plataformas deportivas más valiosas en meses de competiciones importantes

            # Factor 4: Diversidad (complementariedad con otras plataformas ya elegidas)
            diversity_factor = 1.0
            # Si ya hay otras plataformas elegidas para este usuario este mes,
            # valorar más las que ofrecen contenido complementario

            # Combinar factores
            heuristic[plat_idx] = content_price_ratio * continuity_factor * seasonal_factor * diversity_factor

        return heuristic

    def _calculate_potential_content(self, user, platform_name):
        """Calcula el contenido potencial de interés de un usuario para una plataforma, usando sus películas y series."""
        potential_content = 0

        # Procesar películas
        for movie in user.movies:
            movie_interest = movie.get("interest", 1.0)
            movie_platforms = movie.get("platforms", [])

            if platform_name in movie_platforms:
                potential_content += movie_interest * 10  # Valor para película disponible en esa plataforma

        # Procesar series
        for serie in user.series:
            serie_interest = serie.get("interest", 1.0)
            serie_platforms = serie.get("platforms", [])  # Algunas series tienen plataforma a nivel serie

            # Si la serie completa está en la plataforma
            if platform_name in serie_platforms:
                potential_content += serie_interest * 15

            # O si alguna temporada está en esa plataforma
            for temporada in serie.get("season", []):
                temporada_platforms = temporada.get("platforms", [])
                if platform_name in temporada_platforms:
                    potential_content += serie_interest * 10  # Un poco menos que la serie completa

        return max(0.1, potential_content)  # Para evitar cero total

    def _is_pareto_optimal(self, new_objectives: List[float]) -> bool:
        """
        Determina si una solución es Pareto óptima respecto al archivo existente.
        Para minutos ponderados (índice 0): mayor es mejor
        Para costo total (índice 1): menor es mejor
        """
        for _, objectives in self.archive:
            if self._dominates(objectives, new_objectives):
                return False
        return True

    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """
        Determina si obj1 domina a obj2 considerando que:
        - Objetivo 0 (minutos ponderados): MAXIMIZAR (mayor es mejor)
        - Objetivo 1 (costo total): MINIMIZAR (menor es mejor)

        obj1 domina a obj2 si:
        - obj1[0] >= obj2[0] (igual o más minutos)
        - obj1[1] <= obj2[1] (igual o menor costo)
        - Al menos una de las desigualdades es estricta
        """
        # Verificar que es al menos igual en ambos objetivos (recordando que para costo menor es mejor)
        minutos_mejor_o_igual = obj1[0] >= obj2[0]  # Maximizar minutos
        costo_mejor_o_igual = obj1[1] <= obj2[1]  # Minimizar costo

        mejor_o_igual = minutos_mejor_o_igual and costo_mejor_o_igual

        # Verificar que es estrictamente mejor en al menos un objetivo
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
        """Trunca el archivo basado en distancias de crowding para mantener diversidad"""
        distances = self._calculate_crowding_distance()
        sorted_indices = np.argsort(-distances)  # Ordenar por distancia descendente
        self.archive = [self.archive[i] for i in sorted_indices[:self.archive_size]]

    def _calculate_crowding_distance(self) -> np.ndarray:
        """
        Calcula las distancias de crowding para cada solución en el archivo.
        Esto ayuda a mantener la diversidad en el frente de Pareto.
        """
        n_objectives = len(self.archive[0][1])
        n_solutions = len(self.archive)

        if n_solutions <= 2:
            return np.ones(n_solutions) * np.inf

        distances = np.zeros(n_solutions)

        # Extraer valores de fitness para cada objetivo
        fitnesses = np.array([obj for _, obj in self.archive])

        for m in range(n_objectives):
            # Ordenar soluciones por el objetivo actual
            indices = np.argsort(fitnesses[:, m])

            # Asignar distancia infinita a los extremos (para preservarlos)
            distances[indices[0]] = distances[indices[-1]] = np.inf

            # Para cada solución intermedia, calcular distancia normalizada
            if fitnesses[indices[-1], m] - fitnesses[indices[0], m] > 0:  # Evitar división por cero
                norm = fitnesses[indices[-1], m] - fitnesses[indices[0], m]
                for i in range(1, n_solutions - 1):
                    distances[indices[i]] += (fitnesses[indices[i + 1], m] - fitnesses[indices[i - 1], m]) / norm

        return distances

    def _update_pheromone(self, trails: List[Tuple[np.ndarray, List[float]]]):
        """
        Actualiza las feromonas basado en la calidad de las soluciones.
        Usa el concepto de ranking para dar más feromonas a mejores soluciones.
        """
        # Evaporación global
        self.pheromone *= (1 - self.rho)

        # Normalizar los objetivos para comparación justa
        minutos_values = np.array([obj[0] for _, obj in trails])
        costo_values = np.array([obj[1] for _, obj in trails])

        # Evitar división por cero
        minutos_range = max(0.001, np.max(minutos_values) - np.min(minutos_values))
        costo_range = max(0.001, np.max(costo_values) - np.min(costo_values))

        # Normalizar y crear un score combinado (mayor es mejor para minutos, menor es mejor para costo)
        normalized_scores = []
        for _, obj in trails:
            norm_minutos = (obj[0] - np.min(minutos_values)) / minutos_range
            norm_costo = 1 - ((obj[1] - np.min(costo_values)) / costo_range)  # Invertir para minimización

            # Score combinado (promedio simple para este ejemplo)
            combined_score = (norm_minutos + norm_costo) / 2
            normalized_scores.append(combined_score)

        # Ordenar soluciones por score
        ranked_indices = np.argsort(normalized_scores)[::-1]  # Ordenar descendente

        # Actualizar feromonas con más peso para mejores soluciones
        for rank, idx in enumerate(ranked_indices):
            solution, _ = trails[idx]

            # Factor de refuerzo basado en ranking (mejor solución = más refuerzo)
            reinforcement = 1.0 / (rank + 1)

            for month in range(self.n_months):
                for user in range(self.n_users):
                    sol_idx = month * self.n_users + user
                    platform = solution[sol_idx]
                    platform_idx = self.platform_options.index(platform)

                    # Añadir feromonas con base en el ranking
                    self.pheromone[month, user, platform_idx] += reinforcement


def fitness_paco(solution, args):
    n_users = len(args['users'])
    n_months = 12

    candidate = []
    for user_idx in range(n_users):
        user_platforms = []
        for month in range(n_months):
            index = month * n_users + user_idx
            user_platforms.append(solution[index])
        candidate.append(user_platforms)

    minutos_ponderados = calcular_minutos_ponderados(candidate, args)
    costo_total = calcular_costo_total(candidate, args)

    return [minutos_ponderados, costo_total]