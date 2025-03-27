import random
from time import time
import json
import matplotlib.pyplot as plt

import inspyred
from inspyred import ec
from inspyred.ec import emo
from Loaders.LoadStreamingPlans import load_streaming_plan_json
from Loaders.LoadPlatforms import load_platforms_json
from Loaders.LoadUsers import load_users_from_json
from scripts.User_generator import update_users_json

streamingPlans = load_streaming_plan_json("Data/streamingPlans.json")
users = load_users_from_json("Data/users.json")

# Cargar películas por plataforma
with open("Data/MoviesPlatform.json", "r") as f:
    movies_by_platform = json.load(f)

with open("Data/SeriesPlatform.json", "r", encoding="utf-8") as f:
    series_by_platform = json.load(f)

with open("Data/indice_plataformas.json", "r", encoding="utf-8") as f:
    platforms_indexed = json.load(f)

platforms_reverse_index = {v: int(k) for k, v in platforms_indexed.items()}

# Global variables for tracking evolution
generations = []
best_minutes = []
best_cost = []
usuarios_meses = {}


def generar_individuo(random, args):
    individuo = [random.randint(1, 17) for _ in range(12)]
    print(f"Individuos: {individuo}")
    return individuo

def get_platform_name(platform_id):
    """
    Obtiene el nombre de la plataforma a partir de su ID.
    """
    for id_plat, nombre in platforms_indexed.items():
        if id_plat == platform_id:
            return nombre
    return "Desconocida"


def calcular_minutos_ponderados(candidate, args):
    """
    Maximiza el valor total de minutos vistos ponderados por el interés del usuario.

    Parámetros:
    - candidate: Lista de plataformas seleccionadas para cada mes
    - args: Argumentos adicionales (no utilizados actualmente)

    Retorna:
    - float: Suma total de minutos ponderados por interés para todos los usuarios
    """
    minutos_totales_ponderados = 0
    plataformas_por_mes = candidate  # Lista de IDs de plataforma seleccionadas para cada mes

    # Desactivar print de depuración en producción
    verbose = False
    if verbose:
        print("\n--- Evaluando minutos ponderados ---")

    for user in users:
        minutos_disponibles = user.monthly_minutes
        contenidos_disponibles = []

        if verbose:
            print(f"Usuario: {user.name}, Minutos disponibles: {minutos_disponibles}")

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
                    'eficiencia': pelicula['interest']  # Simplificado a solo interés cuando duracion > 0
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
                        'eficiencia': serie['interest']  # Simplificado a solo interés cuando duracion > 0
                    })

        # Ordenar contenidos primero por eficiencia (interés) y luego por duración para desempatar
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

                        if verbose:
                            print(
                                f"  Mes {mes}: Viendo {contenido['id']} - {contenido['duracion']} min, valor: {contenido['valor_ponderado']}")

                        break

    if verbose:
        print(f"Total minutos ponderados: {minutos_totales_ponderados}")

    return minutos_totales_ponderados


def calcular_costo_total(candidate, args):
    """
    Calcula el costo total de suscripciones agrupando usuarios en los planes más baratos posibles,
    evaluando todas las combinaciones posibles para encontrar la solución óptima global.
    """
    costo_total = 0
    plataformas_por_mes = [str(p) for p in candidate]  # Convertir IDs a strings
    print("\n--- Calculando costo total ---")

    for mes in range(12):
        plataforma_id = plataformas_por_mes[mes]

        if plataforma_id not in streamingPlans:
            print(f"Mes {mes + 1}, Plataforma {plataforma_id}: ❌ No en streamingPlans.json")
            continue  # Evita error si la plataforma no está en el JSON

        # Contar cuántos usuarios necesitan esta plataforma en este mes
        usuarios_requieren = sum(
            1 for user in users if any(
                str(plataforma_id) in map(str, pelicula.get('platforms', [])) for pelicula in user.movies
            ) or any(
                str(plataforma_id) in map(str, temporada.get('platforms', []))
                for serie in user.series for temporada in serie.get('season', [])
            )
        )

        if usuarios_requieren == 0:
          #  print(f"Mes {mes + 1}, Plataforma {plataforma_id}: ⚠️ Ningún usuario requiere esta plataforma.")
            continue  # No necesitamos pagar nada por esta plataforma

        # Obtener planes disponibles para esta plataforma
        planes_info = streamingPlans[plataforma_id]
        planes = planes_info if isinstance(planes_info, list) else [planes_info]  # Asegurar lista

      #  print(f"Mes {mes + 1}, Plataforma {plataforma_id}: 🧑‍💻 {usuarios_requieren} usuarios")

        # Encontrar la combinación óptima de planes usando programación dinámica
        # Primero convertimos los planes a una estructura más fácil de usar
        planes_compactos = [(p["perfiles"], p["precio"]) for p in planes]

        # Ordenamos los planes por eficiencia (perfiles/precio) descendente
        planes_compactos.sort(key=lambda p: p[0] / p[1], reverse=True)

        # Función para encontrar la combinación óptima usando programación dinámica
        def encontrar_combinacion_optima(usuarios_a_cubrir):
            # Inicializar DP array
            # dp[i] = (costo mínimo para cubrir i usuarios, planes usados)
            dp = [(float('inf'), []) for _ in range(usuarios_a_cubrir + 1)]
            dp[0] = (0, [])  # Base case: no cuesta nada cubrir 0 usuarios

            # Para cada número de usuarios posible
            for i in range(1, usuarios_a_cubrir + 1):
                # Probar cada plan disponible
                for perfiles, precio in planes_compactos:
                    # Si este plan puede cubrir a los usuarios actuales o más
                    if i <= perfiles:
                        # Si es más barato usar este plan único que la mejor solución previa
                        if precio < dp[i][0]:
                            dp[i] = (precio, [(1, perfiles, precio)])
                    else:
                        # Combinar este plan con la mejor solución para los usuarios restantes
                        usuarios_restantes = i - perfiles
                        costo_combinado = precio + dp[usuarios_restantes][0]

                        if costo_combinado < dp[i][0]:
                            # Crear una nueva lista de planes
                            nuevos_planes = dp[usuarios_restantes][1].copy()

                            # Buscar si ya tenemos este plan en la solución
                            encontrado = False
                            for idx, (cantidad, perf, prec) in enumerate(nuevos_planes):
                                if perf == perfiles and prec == precio:
                                    nuevos_planes[idx] = (cantidad + 1, perf, prec)
                                    encontrado = True
                                    break

                            if not encontrado:
                                nuevos_planes.append((1, perfiles, precio))

                            dp[i] = (costo_combinado, nuevos_planes)

            return dp[usuarios_a_cubrir]

        # Obtenemos la combinación óptima y su costo
        costo_mejor, combinacion_mejor = encontrar_combinacion_optima(usuarios_requieren)

        # Formatear la lista de suscripciones para mostrar
        suscripciones_utilizadas = [f"{num}x({perfiles} perfiles, {precio}€/mes)"
                                    for num, perfiles, precio in combinacion_mejor]

    #    print(f"  ✅ Usamos: {', '.join(suscripciones_utilizadas)} → Costo: {costo_mejor}€")
     #   print(f"  🔹 Costo total para plataforma {plataforma_id} en mes {mes + 1}: {costo_mejor}€")

        costo_total += costo_mejor

    print(f"\n✅ **Costo total final: {costo_total}**")

    return costo_total


def evaluator(candidates, args):
    """
    Evaluador del algoritmo multiobjetivo.
    Devuelve una lista de valores de fitness para cada candidato.
    """
    fitness = []

    print(f"\n--- Evaluando {len(candidates)} individuos ---")

    for candidate in candidates:
        print(f"Evaluando individuo: {candidate}")

        minutos_ponderados = calcular_minutos_ponderados(candidate, args)
        costo_total = calcular_costo_total(candidate, args)

        fitness.append(emo.Pareto([-minutos_ponderados, costo_total]))

    print(f"✅ Evaluación completada: {len(fitness)} soluciones generadas")
    return fitness

def last_generation_update(population, num_generations, args):
    """
    Registra las películas y series vistas por cada usuario en la última generación y actualiza `users.json`.
    """
    if num_generations == args.get('max_generations', 100) - 1:
        print("\n📌 **Registrando películas y series vistas en la última generación...**")

        users_data = args.get('users', [])

        # Tomamos el mejor individuo de la última generación
        best_solution = sorted(population, key=lambda ind: ind.fitness)[0]
        plataformas_por_mes = best_solution.candidate  # La configuración final de plataformas

        watched_movies = {}  # Almacena qué películas vio cada usuario
        watched_series = {}  # Almacena qué series vio cada usuario

        # Procesar cada usuario
        for user in users_data:
            user_id = user.get("user_id")  # ID del usuario
            watched_movies[user_id] = []
            watched_series[user_id] = []

            # Ver qué películas ha visto el usuario
            for movie in user["movies"]:
                for mes, plataforma in enumerate(plataformas_por_mes):
                    if plataforma in movie.get("platforms", []):
                        watched_movies[user_id].append(movie["title"])
                        break  # Si ya la vio, no es necesario seguir revisando meses

            # Ver qué series ha visto el usuario
            for serie in user["series"]:
                for temporada in serie.get("season", []):
                    for mes, plataforma in enumerate(plataformas_por_mes):
                        if plataforma in temporada.get("platforms", []):
                            watched_series[user_id].append(serie["title"])
                            break  # Si ya vio una temporada, consideramos la serie vista

        # 🔥 Guardar los datos en `users.json`
        update_users_json(users_data, watched_movies, watched_series)





def observer(population, num_generations, num_evaluations, args):
    """
    Muestra información de cada generación en la evolución del algoritmo.
    """
    global generations, best_minutes, best_cost, usuarios_meses

    print(f"ESTO ES LA POPULATION: {population} ")

    if num_generations == args.get('max_generations', 100) - 1:
        last_generation_update(population, num_generations, args)

    print(f"\n=== Generación {num_generations} ===")
    print(f"Número de evaluaciones: {num_evaluations}")

    # Extraer fitness
    fitness_values = [ind.fitness for ind in population]

    total_minutos = sum(-fitness[0] for fitness in fitness_values)
    total_costo = sum(fitness[1] for fitness in fitness_values)

    # Guardar en la evolución
    evolucion_minutos.append(total_minutos)
    evolucion_costo.append(total_costo)

    # Mejores y peores valores
    mejor_minutos = min(fitness[0] for fitness in fitness_values)
    peor_minutos = max(fitness[0] for fitness in fitness_values)
    mejor_costo = min(fitness[1] for fitness in fitness_values)
    peor_costo = max(fitness[1] for fitness in fitness_values)

    # Actualizar las listas globales para la gráfica
    generations.append(num_generations)
    best_minutes.append(-mejor_minutos)  # Negamos porque estamos minimizando el negativo
    best_cost.append(mejor_costo)

    print(f"  Mejor Minutos Ponderados: {-mejor_minutos:.2f}, Peor: {-peor_minutos:.2f}")
    print(f"  Mejor Costo Total: {mejor_costo:.2f}, Peor: {peor_costo:.2f}")

    # Mostrar todos los individuos de la población
    print("\n--- Todos los individuos ---")
    for i in range(len(population)):
        minutos_ponderados = -population[i].fitness[0]
        costo_total = population[i].fitness[1]
        print(f"Individuo {i + 1}: Minutos ponderados: {minutos_ponderados:.2f}, Costo total: {costo_total:.2f}")
        print(f"  Configuración: {population[i].candidate}")

        # Almacenar la información del historial de plataformas por usuario
        for mes, plataforma in enumerate(population[i].candidate):
            if plataforma not in usuarios_meses:
                usuarios_meses[plataforma] = {}

            # Iterar sobre los usuarios reales en lugar de usar args['users']
            for user in users:
                user_id = user.id if hasattr(user, 'id') else user.name  # Ajustar según la estructura real de usuario

                if user_id not in usuarios_meses[plataforma]:
                    usuarios_meses[plataforma][user_id] = []

                usuarios_meses[plataforma][user_id].append(mes + 1)  # Guardar el mes donde usó la plataforma


    # Al finalizar la evolución, mostrar el resumen de uso de plataformas por usuario
    if num_generations == args.get('max_generations', 0) - 1:
        print("\n📊 **Resumen de uso de plataformas por usuario**")
        for plataforma, usuarios in usuarios_meses.items():
            print(f"\n🔹 **Plataforma {plataforma}:**")
            for user_id, meses in usuarios.items():
                meses_str = ", ".join(map(str, sorted(set(meses))))
                print(f"  🧑 Usuario {user_id} usó la plataforma en los meses: {meses_str}")





def plot_evolution():
    """ Genera gráficos de la evolución del algoritmo """
    global generations, best_minutes, best_cost

    if not generations:
        print("No hay datos para graficar. Asegúrate de que el algoritmo haya ejecutado al menos una generación.")
        return

    plt.figure(figsize=(12, 6))

    # Gráfico de minutos ponderados
    plt.subplot(1, 2, 1)
    plt.plot(generations, best_minutes, marker='o', linestyle='-', color='b')
    plt.xlabel("Generación")
    plt.ylabel("Minutos Ponderados")
    plt.title("Evolución de Minutos Ponderados")
    plt.grid(True)

    # Gráfico de costo total
    plt.subplot(1, 2, 2)
    plt.plot(generations, best_cost, marker='o', linestyle='-', color='r')
    plt.xlabel("Generación")
    plt.ylabel("Costo Total")
    plt.title("Evolución del Costo Total")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('evolucion.png')  # Guardar gráfico en un archivo
    plt.show()


def plot_pareto_front(algorithm):
    """
      Dibuja el Frente de Pareto final después de la evolución, con puntos unidos.
    """
    pareto_solutions = algorithm.archive
    minutos_ponderados = [-solution.fitness[0] for solution in pareto_solutions]  # Maximizar
    costo_total = [solution.fitness[1] for solution in pareto_solutions]  # Minimizar

    # Ordenar los puntos por costo para que la línea sea continua
    pareto_data = sorted(zip(costo_total, minutos_ponderados))
    costo_total_sorted, minutos_ponderados_sorted = zip(*pareto_data)

    plt.figure(figsize=(8,6))
    plt.plot(costo_total_sorted, minutos_ponderados_sorted, marker='o', linestyle='-', color='red', label="Frente de Pareto")
    plt.xlabel("Costo total (€)")
    plt.ylabel("Minutos ponderados vistos")
    plt.title("Frente de Pareto de las Soluciones")
    plt.legend()
    plt.grid(True)
    plt.savefig("pareto.png")
    plt.show()


def plot_generation_improve():
    """
    Genera gráficos mostrando la evolución del costo total y los minutos ponderados en el algoritmo.
    """
    generaciones = list(range(len(evolucion_minutos)))  # X-Axis

    plt.figure(figsize=(10, 5))

    # 🔹 Gráfico de Minutos Ponderados
    plt.subplot(1, 2, 1)
    plt.plot(generaciones, evolucion_minutos, marker='o', linestyle='-', color='blue', label="Minutos Ponderados")
    plt.xlabel("Generación")
    plt.ylabel("Total Minutos Ponderados")
    plt.title("Evolución de Minutos Ponderados")
    plt.legend()
    plt.grid(True)

    # 🔹 Gráfico de Costo Total
    plt.subplot(1, 2, 2)
    plt.plot(generaciones, evolucion_costo, marker='o', linestyle='-', color='red', label="Costo Total")
    plt.xlabel("Generación")
    plt.ylabel("Costo Total (€)")
    plt.title("Evolución del Costo Total")
    plt.legend()
    plt.grid(True)

    plt.savefig("evolution_improve.png")
    plt.tight_layout()
    plt.show()

# Llamar a la función después de la evolución

evolucion_minutos = []
evolucion_costo = []

def main():
    global generations, best_minutes, best_cost, usuarios_meses

    # Reset global tracking variables
    generations = []
    best_minutes = []
    best_cost = []
    usuarios_meses = {}

    seed = time()
    print(f"Using random seed: {seed}")
    prng = random.Random(seed)
    algorithm = inspyred.ec.emo.NSGA2(prng)
    bounder = inspyred.ec.Bounder(1, len(platforms_indexed))

    # Explicitly configure all required components
    algorithm.selector = inspyred.ec.selectors.tournament_selection
    algorithm.replacer = inspyred.ec.replacers.nsga_replacement
    algorithm.variator = [
        inspyred.ec.variators.uniform_crossover,
        inspyred.ec.variators.random_reset_mutation

    ]
    algorithm.observer = observer
    algorithm.terminator = [inspyred.ec.terminators.generation_termination]

    print("⏳ Iniciando evolución...")

    # Set explicit parameters
    max_gen = 100
    pop_size = 15

    # Prepare arguments dictionary
    args = {
        'platforms_indexed': platforms_indexed,
        'users': users,
        'streamingPlans': streamingPlans,
        'max_generations': max_gen,
        'pop_size': pop_size
    }

    final_pop = algorithm.evolve(
        generator=generar_individuo,
        evaluator=evaluator,
        bounder=bounder,
        pop_size=pop_size,
        maximize=False,
        max_generations=max_gen,
        num_selected=pop_size,  # Select all individuals for potential reproduction
        tournament_size=3,
        num_elites=2,
        mutation_rate=0.9,  # Higher mutation rate for more exploration
        crossover_rate=0.5,
        gaussian_stdev=1.0,  # Higher standard deviation for more diverse mutations
        args=args
    )


    print("✅ Evolución completada.")


    plot_evolution()
    plot_pareto_front(algorithm)
    plot_generation_improve()


if __name__ == "__main__":
    main()



