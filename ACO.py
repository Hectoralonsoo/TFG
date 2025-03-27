import numpy as np
import json
import random
from collections import defaultdict
from Loaders.LoadStreamingPlans import load_streaming_plan_json
from Loaders.LoadPlatforms import load_platforms_json
from Loaders.LoadUsers import load_users_from_json

streamingPlans = load_streaming_plan_json("Data/streamingPlans.json")
users = load_users_from_json("Data/users.json")

# Cargar películas por plataforma
with open("Data/MoviesPlatform.json", "r") as f:
    movies_by_platform = json.load(f)

with open("Data/SeriesPlatform.json", "r", encoding="utf-8") as f:
    series_by_platform = json.load(f)

with open("Data/indice_plataformas.json", "r", encoding="utf-8") as f:
    platforms_indexed = json.load(f)


class AntColonyOptimizer:
    def __init__(self, n_ants, n_iterations, n_months, platform_options, decay=0.5, alpha=1, beta=2):
        """
        Inicializa el optimizador ACO para selección de plataformas.

        Parámetros:
        - n_ants: Número de hormigas en la colonia
        - n_iterations: Número de iteraciones del algoritmo
        - n_months: Número de meses a planificar
        - platform_options: Lista de opciones de plataformas disponibles
        - decay: Factor de evaporación de feromonas
        - alpha: Importancia de las feromonas
        - beta: Importancia de la heurística
        """
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.n_months = n_months
        self.platform_options = platform_options
        self.n_platforms = len(platform_options)
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

        # Inicializar matriz de feromonas
        self.pheromones = np.ones((self.n_months, self.n_platforms))

        # Valores de heurística (inicialmente todos iguales)
        self.heuristic = np.ones((self.n_months, self.n_platforms))

        # Almacenar la mejor solución
        self.best_solution = None
        self.best_fitness = -1

    def _select_platform(self, ant, month):
        """Selecciona una plataforma para un mes específico basado en feromonas y heurística."""
        probabilities = self.pheromones[month] ** self.alpha * self.heuristic[month] ** self.beta
        probabilities = probabilities / np.sum(probabilities)

        return np.random.choice(self.platform_options, p=probabilities)

    def _construct_solution(self):
        """Construye una solución completa para todos los meses."""
        solution = []
        for month in range(self.n_months):
            platform = self._select_platform(None, month)
            solution.append(platform)
        return solution

    def _update_pheromones(self, solutions, fitness_values):
        """Actualiza la matriz de feromonas basada en las soluciones y sus valores de fitness."""
        # Evaporación de feromonas
        self.pheromones *= (1 - self.decay)

        # Actualizar feromonas según la calidad de las soluciones
        for solution, fitness in zip(solutions, fitness_values):
            for month, platform in enumerate(solution):
                platform_idx = self.platform_options.index(platform)
                self.pheromones[month, platform_idx] += fitness

    def _update_heuristic(self, solutions, fitness_values):
        """Actualiza la matriz heurística basada en el rendimiento de las soluciones."""
        # Crear un diccionario para almacenar el rendimiento de cada plataforma por mes
        platform_performance = defaultdict(lambda: defaultdict(list))

        for solution, fitness in zip(solutions, fitness_values):
            for month, platform in enumerate(solution):
                platform_idx = self.platform_options.index(platform)
                platform_performance[month][platform_idx].append(fitness)

        # Actualizar la matriz heurística
        for month in range(self.n_months):
            for platform_idx in range(self.n_platforms):
                fitness_values = platform_performance[month][platform_idx]
                if fitness_values:
                    self.heuristic[month, platform_idx] = np.mean(fitness_values)

    def optimize(self, fitness_function):
        """
        Ejecuta el algoritmo ACO para encontrar la mejor combinación de plataformas.

        Parámetros:
        - fitness_function: Función que evalúa la calidad de una solución

        Retorna:
        - La mejor solución encontrada
        """
        print("Iniciando optimización con ACO...")

        for iteration in range(self.n_iterations):
            # Construir soluciones
            solutions = [self._construct_solution() for _ in range(self.n_ants)]

            # Evaluar soluciones
            fitness_values = [fitness_function(solution, {}) for solution in solutions]

            # Encontrar la mejor solución de esta iteración
            best_idx = np.argmax(fitness_values)
            current_best = solutions[best_idx]
            current_best_fitness = fitness_values[best_idx]

            # Actualizar la mejor solución global
            if current_best_fitness > self.best_fitness:
                self.best_solution = current_best
                self.best_fitness = current_best_fitness
                print(f"Iteración {iteration + 1}: Nueva mejor solución encontrada! Fitness: {self.best_fitness}")

            # Actualizar feromonas y heurística
            self._update_pheromones(solutions, fitness_values)
            self._update_heuristic(solutions, fitness_values)

            if (iteration + 1) % 10 == 0:
                print(f"Iteración {iteration + 1}/{self.n_iterations}, Mejor fitness: {self.best_fitness}")

        print("Optimización completada!")
        print(f"Mejor solución: {self.best_solution}")
        print(f"Fitness: {self.best_fitness}")

        return self.best_solution, self.best_fitness


def calcular_minutos_ponderados(candidate, args):
    """
    Función objetivo que calcula los minutos ponderados para una solución candidata.

    Parámetros:
    - candidate: Lista de plataformas seleccionadas para cada mes
    - args: Argumentos adicionales (no utilizados actualmente)

    Retorna:
    - float: Suma total de minutos ponderados por interés para todos los usuarios
    """
    minutos_totales_ponderados = 0
    plataformas_por_mes = candidate

    for user in users:
        minutos_disponibles = user.monthly_minutes
        contenidos_disponibles = []

        # Mapeo de plataformas por mes para búsqueda eficiente
        plataformas_mes_dict = {mes: plat_id for mes, plat_id in enumerate(plataformas_por_mes)}

        # Procesar películas disponibles
        for pelicula in user.movies:
            meses_disponibles = [
                mes for mes, plat_id in plataformas_mes_dict.items()
                if plat_id in pelicula['platforms']
            ]

            duracion = pelicula['movie_duration']
            if duracion <= 0:  # Evitar división por cero y contenidos sin duración
                continue

            if meses_disponibles:
                contenidos_disponibles.append({
                    'tipo': 'pelicula',
                    'id': pelicula['title'],
                    'duracion': duracion,
                    'interes': pelicula['interest'],
                    'valor_ponderado': duracion * pelicula['interest'],
                    'meses': meses_disponibles,
                    'eficiencia': pelicula['interest']
                })

        # Procesar series disponibles
        for serie in user.series:
            # Plataformas donde está disponible la serie completa
            plataformas_serie = serie.get('platforms', [])

            for temporada in serie['season']:
                # Combinar plataformas de la serie y temporada específica
                plataformas_temporada = set(temporada.get('platforms', []) + plataformas_serie)

                duracion = temporada['season_duration']
                if duracion <= 0:  # Evitar división por cero y contenidos sin duración
                    continue

                meses_disponibles = [
                    mes for mes, plat_id in plataformas_mes_dict.items()
                    if plat_id in plataformas_temporada
                ]

                if meses_disponibles:
                    contenidos_disponibles.append({
                        'tipo': 'serie',
                        'id': f"{serie['title']} - T{temporada['season_number']}",
                        'duracion': duracion,
                        'interes': serie['interest'],
                        'valor_ponderado': duracion * serie['interest'],
                        'meses': meses_disponibles,
                        'eficiencia': serie['interest']
                    })

        # Ordenar contenidos por eficiencia (interés) y luego por duración para desempatar
        contenidos_disponibles.sort(key=lambda x: (x['eficiencia'], -x['duracion']), reverse=True)

        # Diccionario para llevar el registro de minutos utilizados por mes
        minutos_usados_por_mes = {mes: 0 for mes in range(len(plataformas_por_mes))}
        contenidos_vistos = set()

        # Asignar contenidos eficientemente
        for contenido in contenidos_disponibles:
            # Ordenar meses por menor uso (para distribuir contenido uniformemente)
            meses_ordenados = sorted(contenido['meses'], key=lambda m: minutos_usados_por_mes[m])

            for mes in meses_ordenados:
                clave_contenido = (contenido['id'], mes)

                if clave_contenido not in contenidos_vistos:
                    # Verificar si hay suficientes minutos disponibles en este mes
                    if minutos_usados_por_mes[mes] + contenido['duracion'] <= minutos_disponibles:
                        minutos_totales_ponderados += contenido['valor_ponderado']
                        minutos_usados_por_mes[mes] += contenido['duracion']
                        contenidos_vistos.add(clave_contenido)
                        break

    return minutos_totales_ponderados


def main():
    """Función principal para ejecutar el algoritmo ACO."""
    # Parámetros del problema
    n_months = 12  # Número de meses a planificar
    # Obtener las plataformas disponibles de los datos
    plataformas_disponibles = obtener_plataformas_disponibles()

    # Parámetros del algoritmo ACO
    n_ants = 10
    n_iterations = 1000
    decay = 0.1
    alpha = 1
    beta = 3

    # Inicializar y ejecutar el algoritmo ACO
    aco = AntColonyOptimizer(
        n_ants=n_ants,
        n_iterations=n_iterations,
        n_months=n_months,
        platform_options=plataformas_disponibles,
        decay=decay,
        alpha=alpha,
        beta=beta
    )

    # Obtener la mejor solución
    mejor_solucion, mejor_fitness = aco.optimize(calcular_minutos_ponderados)

    # Mostrar resultados
    print("\nResultados finales:")
    print(f"Mejor combinación de plataformas:")
    for mes, plataforma in enumerate(mejor_solucion):
        print(f"Mes {mes + 1}: Plataforma {plataforma}")
    print(f"Valor total de minutos ponderados: {mejor_fitness}")


def obtener_plataformas_disponibles():
    """
    Extrae todas las plataformas disponibles a partir de los datos de usuarios.
    """
    plataformas = set()

    for user in users:
        # Extraer plataformas de películas
        for pelicula in user.movies:
            plataformas.update(pelicula['platforms'])

        # Extraer plataformas de series
        for serie in user.series:
            if 'platforms' in serie:
                plataformas.update(serie['platforms'])

            for temporada in serie['season']:
                if 'platforms' in temporada:
                    plataformas.update(temporada['platforms'])

    return list(plataformas)


if __name__ == "__main__":
    main()