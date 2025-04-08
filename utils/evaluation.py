from utils.heuristics import encontrar_combinacion_optima
from inspyred.ec import emo

def calcular_minutos_ponderados(candidate, args):
    """
    Maximiza el valor total de minutos vistos ponderados por el interés del usuario.

    Parámetros:
    - candidate: Lista de plataformas seleccionadas para cada mes
    - args: Argumentos adicionales (no utilizados actualmente)

    Retorna:
    - float: Suma total de minutos ponderados por interés para todos los usuarios
    """

    users = args['users']
    minutos_totales_ponderados = 0

    # Desactivar print de depuración en producción
    verbose = False
    if verbose:
        print("\n--- Evaluando minutos ponderados ---")

    for i, user in enumerate(users):
        plataformas_por_mes = candidate[i]
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
    Calcula el costo total mensual agrupando usuarios por plataforma y
    muestra por consola un resumen claro por cada mes.
    """
    costo_total = 0
    num_users = len(candidate)
    streamingPlans = args["streamingPlans"]
    platforms_indexed = args["platforms_indexed"]
    print(platforms_indexed)

    print("\n📅 === CÁLCULO DE COSTO TOTAL ===")

    for mes in range(12):
        print(f"\n🔸 Mes {mes + 1}:")
        plataformas_mes = [str(candidate[i][mes]) for i in range(num_users)]
        plataformas_contadas = set(plataformas_mes)

        costo_mes = 0

        for plataforma_id in plataformas_contadas:
            if plataforma_id not in streamingPlans:
                continue

            # Obtener usuarios asignados a esta plataforma en este mes
            usuarios_asignados = [
                i for i in range(num_users) if str(candidate[i][mes]) == plataforma_id
            ]
            num_usuarios = len(usuarios_asignados)

            if num_usuarios == 0:
                continue

            # Obtener planes de esa plataforma
            planes_info = streamingPlans[plataforma_id]
            planes = planes_info if isinstance(planes_info, list) else [planes_info]
            planes_compactos = [(p["perfiles"], p["precio"]) for p in planes]
            planes_compactos.sort(key=lambda p: p[0] / p[1], reverse=True)

            # Calcular mejor combinación de planes
            costo_plataforma, combinacion = encontrar_combinacion_optima(num_usuarios, planes_compactos)
            costo_mes += costo_plataforma

            nombre_plataforma = platforms_indexed.get(str(plataforma_id), f"Plataforma {plataforma_id}")
            resumen_planes = ", ".join([f"{cant}x({perf} perfiles, {precio}€)" for cant, perf, precio in combinacion])

            print(f"  {nombre_plataforma} → {resumen_planes} → {num_usuarios} usuario(s)")

        print(f"🔹 Total mes {mes + 1}: {costo_mes:.2f}€")
        costo_total += costo_mes

    print(f"\n✅ Costo total anual: {costo_total:.2f}€")
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


def evaluatorSPEA2(candidates, args):

    results = []

    for candidate in candidates:
        minutos_ponderados = calcular_minutos_ponderados(candidate, args)
        costo_total = calcular_costo_total(candidate, args)

        results.append([costo_total, -minutos_ponderados])

    return results