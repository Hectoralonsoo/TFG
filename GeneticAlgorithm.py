import random
from time import time
import json

import inspyred
from inspyred import ec
from inspyred.ec import emo
from Loaders.LoadStreamingPlans import load_streaming_plan_json
from Loaders.LoadPlatforms import load_platforms_json
from Loaders.LoadUsers import load_users_from_json

# ==============================
# Cargar datos
# ==============================
streamingPlans = load_streaming_plan_json("Data/streamingPlans.json")
users = load_users_from_json("Data/users.json")

# Cargar películas por plataforma
with open("Data/MoviesPlatform.json", "r") as f:
    movies_by_platform = json.load(f)
#print(movies_by_platform)

with open("Data/SeriesPlatform.json", "r") as f:
    series_by_platform = json.load(f)



with open("Data/indice_plataformas.json", "r", encoding="utf-8") as f:
    platforms_indexed = json.load(f)



# ==============================
# Generador de Individuos
# ==============================
def generar_individuo(random, args):
    """
    Genera un individuo representado como una lista de plataformas elegidas para cada mes.
    """
    individuo = []
    for mes in range(12):  # Eliminar el espacio extra antes de 'for'
        plataforma = random.choice(list(platforms_indexed.keys()))
        individuo.append(plataforma)

    print(individuo)
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
    Calcula los minutos totales de contenido vistos por los usuarios, ponderados por el interés.
    Maximiza: Σ(minutos_vistos_ponderados)
    """
    print(f"Candidate: {candidate}")
    print(f"Platforms indexed: {platforms_indexed}")
    minutos_totales_ponderados = 0

    # Mapear el ID de plataforma a su nombre para todos los meses
    plataformas_por_mes = [get_platform_name(platform_id) for platform_id in candidate]

    # Para cada usuario, calcular cuántos minutos puede ver ponderados por interés
    for user in users:
        minutos_disponibles = user.monthly_minutes
        total_minutos_usuario = 0

        # Lista para almacenar todos los contenidos disponibles para este usuario
        contenidos_disponibles = []

        # Procesar películas disponibles
        for pelicula in user.movies:
            # Verificar en qué meses está disponible esta película
            meses_disponibles = []
            for mes, plataforma in enumerate(plataformas_por_mes):
                if plataforma in pelicula.get('platforms', []):
                    meses_disponibles.append(mes)

            if meses_disponibles:
                contenidos_disponibles.append({
                    'tipo': 'pelicula',
                    'titulo': pelicula.get('title'),
                    'duracion': pelicula.get('movie_duration'),
                    'interes': pelicula.get('interest'),
                    'valor_ponderado': pelicula.get('movie_duration') * pelicula.get('interest'),
                    'meses': meses_disponibles
                })

        # Procesar series disponibles
        for serie in user.series:
            for temporada in serie.get('season', []):
                # Verificar en qué meses está disponible esta temporada
                meses_disponibles = []
                for mes, plataforma in enumerate(plataformas_por_mes):
                    if plataforma in temporada.get('platforms', []) or plataforma in serie.get('platforms', []):
                        meses_disponibles.append(mes)

                if meses_disponibles:
                    contenidos_disponibles.append({
                        'tipo': 'serie',
                        'titulo': f"{serie.get('title', '')} - {temporada.get('season_name', '')}",
                        'duracion': temporada.get('season_duration', 300),
                        'interes': serie.get('interest', 0.5),
                        'valor_ponderado': temporada.get('season_duration', 300) * serie.get('interest', 0.5),
                        'meses': meses_disponibles
                    })

        # Ordenar contenidos por valor ponderado (interés * duración) en orden descendente
        contenidos_disponibles.sort(key=lambda x: x['valor_ponderado'], reverse=True)

        # Seguimiento de qué contenidos ya se han visto y en qué mes
        contenidos_vistos = set()  # Conjunto de tuplas (título, mes)

        # El usuario verá los contenidos de mayor valor hasta agotar su tiempo disponible
        for contenido in contenidos_disponibles:
            # Solo considerar el contenido si no se ha visto antes y está disponible en algún mes
            for mes in contenido['meses']:
                clave_contenido = (contenido['titulo'], mes)
                if clave_contenido not in contenidos_vistos and total_minutos_usuario + contenido[
                    'duracion'] <= minutos_disponibles:
                    # El usuario puede ver este contenido completamente
                    minutos_totales_ponderados += contenido['valor_ponderado']
                    total_minutos_usuario += contenido['duracion']
                    contenidos_vistos.add(clave_contenido)
                    break  # Solo ver el contenido una vez

    return minutos_totales_ponderados


def calcular_costo_total(candidate, args):
    """
    Calcula el costo total optimizado de las suscripciones para los 12 meses.
    Analiza cuántos usuarios necesitan cada plataforma en cada mes y los agrupa
    de la manera más rentable posible.
    """
    # Extraer las variables necesarias del objeto args
    platforms_indexed = args.get('platforms_indexed', {})
    users = args.get('users', [])
    streamingPlans = args.get('streamingPlans', [])

    # Debug para ver qué valores se están recibiendo
    print(f"Candidate: {candidate}")
    print(f"Platforms indexed: {platforms_indexed}")
    print(f"Users count: {len(users)}")
    print(f"StreamingPlans count: {len(streamingPlans)}")

    # Convertir IDs de plataforma a nombres para cada mes
    plataformas_por_mes = []
    for plataforma_id in candidate:
        nombre_plataforma = None
        for nombre, id in platforms_indexed.items():
            if id == plataforma_id:
                nombre_plataforma = nombre
                break
        plataformas_por_mes.append(nombre_plataforma)

    print(f"Plataformas por mes: {plataformas_por_mes}")

    # Cargar información de planes
    planes_plataformas = {}
    try:
        with open("Data/streamingPlans.json", "r", encoding="utf-8") as f:
            planes_plataformas = json.load(f)
        print("Archivo streamingPlans.json cargado correctamente")
    except Exception as e:
        print(f"Error al cargar streamingPlans.json: {e}")
        # Usar los planes de streamingPlans como respaldo
        for plan in streamingPlans:
            platform_id = plan.get('platform_id')
            platform_name = None
            for nombre, id in platforms_indexed.items():
                if id == platform_id:
                    platform_name = nombre
                    break

            if platform_name:
                if platform_name not in planes_plataformas:
                    planes_plataformas[platform_name] = {"planes": []}

                planes_plataformas[platform_name]["planes"].append({
                    "perfiles": plan.get('profiles', 1),
                    "precio": plan.get('monthly_cost', 10.0)
                })

    # Calcular costo total para todos los meses
    costo_total = 0

    # Para cada mes, determinar cuántos usuarios necesitan cada plataforma
    for mes in range(len(candidate)):
        plataforma_mes = plataformas_por_mes[mes]
        if not plataforma_mes:  # Si la plataforma es None, continuar al siguiente mes
            print(f"Plataforma para el mes {mes} no encontrada")
            continue

        usuarios_por_plataforma = {}

        # Contar cuántos usuarios necesitan cada plataforma en este mes
        for usuario in users:
            # Verificar el tipo de usuario y acceder a sus atributos correctamente
            user_id = usuario.get('user_id') if isinstance(usuario, dict) else getattr(usuario, 'user_id', None)

            # Verificar películas
            movies = usuario.get('movies', []) if isinstance(usuario, dict) else getattr(usuario, 'movies', [])
            for pelicula in movies:
                platforms = pelicula.get('platforms', [])
                if plataforma_mes in platforms:
                    if plataforma_mes not in usuarios_por_plataforma:
                        usuarios_por_plataforma[plataforma_mes] = set()
                    usuarios_por_plataforma[plataforma_mes].add(user_id)

            # Verificar series
            series = usuario.get('series', []) if isinstance(usuario, dict) else getattr(usuario, 'series', [])
            for serie in series:
                # Verificar si la plataforma está en la serie directamente
                if plataforma_mes in serie.get('platforms', []):
                    if plataforma_mes not in usuarios_por_plataforma:
                        usuarios_por_plataforma[plataforma_mes] = set()
                    usuarios_por_plataforma[plataforma_mes].add(user_id)
                    continue

                # Verificar cada temporada
                for temporada in serie.get('season', []):
                    if plataforma_mes in temporada.get('platforms', []):
                        if plataforma_mes not in usuarios_por_plataforma:
                            usuarios_por_plataforma[plataforma_mes] = set()
                        usuarios_por_plataforma[plataforma_mes].add(user_id)
                        break

        # Calcular el costo óptimo para cada plataforma en este mes
        costo_mes = 0
        for plataforma, usuarios in usuarios_por_plataforma.items():
            num_usuarios = len(usuarios)
            print(f"Mes {mes}, Plataforma {plataforma}: {num_usuarios} usuarios")

            if plataforma in planes_plataformas:
                mejor_costo = float('inf')

                # Caso especial: si la plataforma tiene un solo plan
                if "plan" in planes_plataformas[plataforma]:
                    plan = planes_plataformas[plataforma]["plan"]
                    perfiles_por_plan = plan.get("perfiles", 1)
                    precio_plan = plan.get("precio", 0)

                    # Calcular cuántas suscripciones necesitamos
                    if perfiles_por_plan > 0:
                        num_suscripciones = (num_usuarios + perfiles_por_plan - 1) // perfiles_por_plan
                        costo = num_suscripciones * precio_plan
                        mejor_costo = costo
                        print(f"  Plan único: {num_suscripciones} suscripciones x {precio_plan} = {costo}")

                # Caso de múltiples planes: probar todas las combinaciones
                elif "planes" in planes_plataformas[plataforma]:
                    # Primera estrategia: usar solo un tipo de plan
                    for plan in planes_plataformas[plataforma]["planes"]:
                        perfiles_por_plan = plan.get("perfiles", 1)
                        precio_plan = plan.get("precio", 0)

                        # Calcular cuántas suscripciones necesitamos
                        if perfiles_por_plan > 0:
                            num_suscripciones = (num_usuarios + perfiles_por_plan - 1) // perfiles_por_plan
                            costo = num_suscripciones * precio_plan
                            print(f"  Plan múltiple: {num_suscripciones} x {precio_plan} = {costo}")

                            if costo < mejor_costo:
                                mejor_costo = costo

                    # Segunda estrategia: combinar planes (podría ser más complejo)
                    # Por simplicidad, probamos combinaciones básicas
                    planes = sorted(planes_plataformas[plataforma]["planes"],
                                    key=lambda x: x.get("precio", 0) / max(1, x.get("perfiles", 1)))

                    usuarios_restantes = num_usuarios
                    costo_combinado = 0

                    for plan in planes:
                        # Usar el plan más eficiente primero
                        perfiles = plan.get("perfiles", 1)
                        precio = plan.get("precio", 0)

                        if perfiles > 0:
                            num_planes = usuarios_restantes // perfiles
                            if usuarios_restantes % perfiles > 0 and usuarios_restantes > 0:
                                num_planes += 1

                            costo_combinado += num_planes * precio
                            usuarios_restantes -= num_planes * perfiles
                            print(f"  Combinación: {num_planes} x {precio} (perfiles: {perfiles})")

                        if usuarios_restantes <= 0:
                            break

                    print(f"  Costo combinado: {costo_combinado}")
                    if costo_combinado < mejor_costo:
                        mejor_costo = costo_combinado

                if mejor_costo != float('inf'):
                    costo_mes += mejor_costo
                    print(f"  Mejor costo para {plataforma}: {mejor_costo}")
                else:
                    print(f"  No se pudo calcular costo para {plataforma}")
            else:
                # Si no tenemos información del plan, usar un costo predeterminado de streamingPlans
                print(f"  No hay información de planes para {plataforma}, usando streamingPlans")
                found_plan = False
                for plan in streamingPlans:
                    plan_id = None
                    for nombre, id in platforms_indexed.items():
                        if nombre == plataforma:
                            plan_id = id
                            break

                    if plan.get('platform_id') == plan_id:
                        found_plan = True
                        # Calcular cuántas suscripciones necesitamos
                        perfiles = plan.get('profiles', 1)
                        costo = 0
                        if perfiles > 0:
                            num_suscripciones = (num_usuarios + perfiles - 1) // perfiles
                            costo = num_suscripciones * plan.get('monthly_cost', 10.0)
                        costo_mes += costo
                        print(
                            f"  Usando plan predeterminado: {num_suscripciones} x {plan.get('monthly_cost', 10.0)} = {costo}")

                if not found_plan:
                    print(f"  No se encontró plan para {plataforma} en streamingPlans")

        print(f"Costo del mes {mes}: {costo_mes}")
        costo_total += costo_mes

    print(f"Costo total: {costo_total}")
    return costo_total





def evaluator(candidates, args):
    """
    Evaluador para algoritmo multiobjetivo (emo.Pareto).
    Calcula dos objetivos para cada candidato:
    1. Minutos ponderados (a maximizar)
    2. Costo total (a minimizar)

    Retorna una lista de listas donde cada sublista contiene los valores
    de los objetivos para cada candidato.
    """
    fitness = []

    for candidate in candidates:
        # Calcular minutos ponderados (a maximizar)
        minutos_ponderados = calcular_minutos_ponderados(candidate, args)

        # Calcular costo total (a minimizar)
        costo_total = calcular_costo_total(candidate, args)

        # Para emo.Pareto, retornamos una lista de objetivos
        # Importante: en emo.Pareto, por defecto se MINIMIZA todos los objetivos
        # Por eso multiplicamos los minutos ponderados por -1 para maximizarlos
        fitness.append([-minutos_ponderados, costo_total])
        print("Fitness values:", fitness)

    return fitness


# ==============================
# Configuración y Ejecución del Algoritmo Genético
# ==============================
def main():
    seed = time()
    prng = random.Random(seed)

    # Uso del algoritmo Pareto para optimización multiobjetivo
    algorithm = inspyred.ec.emo.NSGA2(prng)

    # Límites para los valores del individuo (plataformas)
    bounder = inspyred.ec.Bounder(1, 18)  # Ajusta según el número de plataformas

    # Configuración de operadores
    algorithm.variator = [
        inspyred.ec.variators.uniform_crossover,
        inspyred.ec.variators.gaussian_mutation
    ]

    def multi_objective_stats_observer(population, num_generations, num_evaluations, args):
        """Custom observer that handles multi-objective fitness values"""
        if num_generations > 0:
            print('Generation {0}: Archive size = {1}'.format(num_generations,
                                                              len(args['_ec'].archive)))

            # You can add more stats if needed
            if len(args['_ec'].archive) > 0:
                print('  First objective range: [{0:.2f}, {1:.2f}]'.format(
                    min(ind.fitness[0] for ind in args['_ec'].archive),
                    max(ind.fitness[0] for ind in args['_ec'].archive)))
                print('  Second objective range: [{0:.2f}, {1:.2f}]'.format(
                    min(ind.fitness[1] for ind in args['_ec'].archive),
                    max(ind.fitness[1] for ind in args['_ec'].archive)))

    # Then use this observer
    algorithm.observer = [
        multi_objective_stats_observer,
        inspyred.ec.observers.archive_observer
    ]


    # Evolución del algoritmo
    final_pop = algorithm.evolve(
        generator=generar_individuo,
        evaluator=evaluator,
        bounder=bounder,
        pop_size=10,  # Mayor población para mejor exploración
        maximize=False,  # Todos los objetivos se minimizan (los minutos ya están negados)
        max_generations=100,
        num_selected=5,  # Número de individuos seleccionados para reproducción
        tournament_size=2,  # Tamaño del torneo para selección
        num_elites=1,  # Número de elites conservados entre generaciones
        mutation_rate=0.1,  # Tasa de mutación
        crossover_rate=0.9,  # Tasa de cruce
        gaussian_stdev=0.1,  # Desviación estándar para mutación gaussiana
        args={'platforms_indexed': platforms_indexed, 'users': users, 'streamingPlans': streamingPlans}
    )

    # Obtener el archivo de soluciones no dominadas (frente de Pareto)
    pareto_front = algorithm.archive

    # Imprimir resultados
    print("Número de soluciones en el frente de Pareto:", len(pareto_front))
    print("\nSoluciones no dominadas (frente de Pareto):")

    # Ordenar por minutos ponderados (primer objetivo)
    sorted_front = sorted(pareto_front, key=lambda x: x.fitness[0])

    for i, solution in enumerate(sorted_front):
        # Convertir valores de fitness a sus valores reales
        minutos = -solution.fitness[0]  # Negamos para recuperar el valor positivo
        costo = solution.fitness[1]

        print(f"Solución {i + 1}:")
        print(f"  Minutos ponderados: {minutos:.2f}")
        print(f"  Costo total: ${costo:.2f}")

        # Podemos mostrar también la configuración de plataformas por mes
        print("  Plataformas por mes:")
        for mes, plataforma_id in enumerate(solution.candidate):
            nombre_plataforma = get_platform_name(plataforma_id)
            print(f"    Mes {mes + 1}: {nombre_plataforma} (ID: {plataforma_id})")
        print()

    # Opcional: Guardar las soluciones en un archivo
    guardar_resultados(sorted_front)

    return pareto_front, algorithm.archive


def guardar_resultados(soluciones):
    """
    Guarda las soluciones Pareto en un archivo JSON.
    """
    resultados = []

    for i, solution in enumerate(soluciones):
        # Convertir la solución a un formato más legible
        solucion_dict = {
            "id": i + 1,
            "minutos_ponderados": -solution.fitness[0],
            "costo_total": solution.fitness[1],
            "plataformas_por_mes": []
        }

        # Agregar información de plataformas por mes
        for mes, plataforma_id in enumerate(solution.candidate):
            nombre_plataforma = get_platform_name(plataforma_id)

            solucion_dict["plataformas_por_mes"].append({
                "mes": mes + 1,
                "plataforma_id": int(plataforma_id),
                "plataforma_nombre": nombre_plataforma
            })

        resultados.append(solucion_dict)

    # Guardar en archivo JSON
    with open("resultados_optimizacion.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print("Resultados guardados en 'resultados_optimizacion.json'")


main()