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


streamingPlans = load_streaming_plan_json("Data/streamingPlans.json")
users = load_users_from_json("Data/users.json")

# Cargar películas por plataforma
with open("Data/MoviesPlatform.json", "r") as f:
    movies_by_platform = json.load(f)
#print(movies_by_platform)

with open("Data/SeriesPlatform.json", "r", encoding="utf-8") as f:
    series_by_platform = json.load(f)



with open("Data/indice_plataformas.json", "r", encoding="utf-8") as f:
    platforms_indexed = json.load(f)


platforms_reverse_index = {v: int(k) for k, v in platforms_indexed.items()}

def generar_individuo(random, args):
    return [random.randint(1, 17) for _ in range(12)]

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
    Maximiza: Σ(minutos_vistos_ponderados)
    """
    minutos_totales_ponderados = 0
    plataformas_por_mes = candidate  # Directamente IDs de plataforma
    print("\n--- Evaluando minutos ponderados ---")

    for user in users:
        minutos_disponibles = user.monthly_minutes
        total_minutos_usuario = 0
        contenidos_disponibles = []

        print(f"Usuario: {user.name}, Minutos disponibles: {minutos_disponibles}")

        # Procesar películas disponibles
        for pelicula in user.movies:
            meses_disponibles = [mes for mes, plataforma_id in enumerate(plataformas_por_mes)
                                 if plataforma_id in pelicula['platforms']]
            if meses_disponibles:
                print(f"  Película: {pelicula['title']}, Disponible en meses: {meses_disponibles}")
                contenidos_disponibles.append({
                    'nombre': pelicula['title'],
                    'duracion': pelicula['movie_duration'],
                    'interes': pelicula['interest'],
                    'valor_ponderado': pelicula['movie_duration'] * pelicula['interest'],
                    'meses': meses_disponibles
                })

        # Procesar series disponibles
        for serie in user.series:
            for temporada in serie['season']:
                meses_disponibles = [mes for mes, plataforma_id in enumerate(plataformas_por_mes)
                                     if plataforma_id in temporada['platforms'] or plataforma_id in serie['platforms']]
                if meses_disponibles:
                    print(f"  Serie: {serie['title']}, Temporada: {temporada['season_name']}, Disponible en meses: {meses_disponibles}")
                    contenidos_disponibles.append({
                        'nombre': serie['title'],
                        'season': temporada['season_number'],
                        'duracion': temporada['season_duration'],
                        'interes': serie['interest'],
                        'valor_ponderado': temporada['season_duration'] * serie['interest'],
                        'meses': meses_disponibles
                    })

        # Ordenar contenidos por valor ponderado (interés * duración)
        contenidos_disponibles.sort(key=lambda x: x['valor_ponderado'], reverse=True)

        contenidos_vistos = set()
        for contenido in contenidos_disponibles:
            for mes in contenido['meses']:
                clave_contenido = (contenido['nombre'], mes)
                if clave_contenido not in contenidos_vistos and total_minutos_usuario + contenido['duracion'] <= minutos_disponibles:
                    minutos_totales_ponderados += contenido['valor_ponderado']
                    total_minutos_usuario += contenido['duracion']
                    contenidos_vistos.add(clave_contenido)
                    break

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
            print(f"Mes {mes + 1}, Plataforma {plataforma_id}: ⚠️ Ningún usuario requiere esta plataforma.")
            continue  # No necesitamos pagar nada por esta plataforma

        # Obtener planes disponibles para esta plataforma
        planes_info = streamingPlans[plataforma_id]
        planes = planes_info if isinstance(planes_info, list) else [planes_info]  # Asegurar lista

        print(f"Mes {mes + 1}, Plataforma {plataforma_id}: 🧑‍💻 {usuarios_requieren} usuarios")

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

        print(f"  ✅ Usamos: {', '.join(suscripciones_utilizadas)} → Costo: {costo_mejor}€")
        print(f"  🔹 Costo total para plataforma {plataforma_id} en mes {mes + 1}: {costo_mejor}€")

        costo_total += costo_mejor

    print(f"\n✅ **Costo total final: {costo_total}**")
    return costo_total





def evaluator(candidates, args):
    """
    Evaluador para calcular fitness de cada candidato.
    """
    fitness = []
    for candidate in candidates:
        minutos_ponderados = calcular_minutos_ponderados(candidate, args)
        costo_total = calcular_costo_total(candidate, args)
        print(f"Evaluando candidato: {candidate}")
        print(f"  -> Minutos ponderados: {minutos_ponderados}, Costo total: {costo_total}")
        fitness.append([-minutos_ponderados, costo_total])
    return fitness

generations = []
best_minutes = []
best_cost = []

def observer(population, num_generations, num_evaluations, args):
    """ Observer para registrar y graficar la evolución """
    generations.append(num_generations)

    # Encontrar el mejor individuo en términos de minutos ponderados
    best_individual = max(population, key=lambda ind: -ind.fitness[0])
    best_minutes.append(-best_individual.fitness[0])  # Negamos porque queremos maximizar
    best_cost.append(best_individual.fitness[1])

    print(f"\n=== 🔄 Generación {num_generations} ===")
    print(f"Evaluaciones: {num_evaluations}")
    print(f"🏆 Mejor Minutos Ponderados: {best_minutes[-1]:.2f}, Costo Total: {best_cost[-1]:.2f}")


def plot_evolution():
    """ Genera gráficos de la evolución del algoritmo """
    plt.figure(figsize=(10, 5))

    # Gráfico de minutos ponderados
    plt.subplot(1, 2, 1)
    plt.plot(generations, best_minutes, marker='o', linestyle='-', color='b')
    plt.xlabel("Generación")
    plt.ylabel("Minutos Ponderados")
    plt.title("Evolución de Minutos Ponderados")

    # Gráfico de costo total
    plt.subplot(1, 2, 2)
    plt.plot(generations, best_cost, marker='o', linestyle='-', color='r')
    plt.xlabel("Generación")
    plt.ylabel("Costo Total")
    plt.title("Evolución del Costo Total")

    plt.tight_layout()
    plt.show()

def main():
    seed = time()
    prng = random.Random(seed)
    algorithm = inspyred.ec.emo.NSGA2(prng)
    bounder = inspyred.ec.Bounder(1, len(platforms_indexed))
    algorithm.replacer = inspyred.ec.replacers.crowding_replacement
    algorithm.variator = [
        inspyred.ec.variators.uniform_crossover,
        inspyred.ec.variators.gaussian_mutation
    ]
    algorithm.observer = [observer]

    print("⏳ Iniciando evolución...")

    final_pop = algorithm.evolve(
        generator=generar_individuo,
        evaluator=evaluator,
        bounder=bounder,
        pop_size=200,
        max_generations=100,
        num_selected=50,
        tournament_size=5,
        num_elites=5,
        mutation_rate=0.2,
        crossover_rate=0.9,
   #  gaussian_stdev=0.1,
   #     args={'platforms_indexed': platforms_indexed, 'users': users, 'streamingPlans': streamingPlans}
    )

    plot_evolution()

    print("✅ Evolución completada.")

    print("Evolución finalizada. Soluciones en el frente de Pareto:")
    for solution in algorithm.archive:
        print(f"Minutos ponderados: {-solution.fitness[0]}, Costo total: {solution.fitness[1]}")


main()
