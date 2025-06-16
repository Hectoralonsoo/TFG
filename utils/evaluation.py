from utils.heuristics import encontrar_combinacion_optima
from inspyred.ec import emo


def calcular_minutos_ponderados(candidate, args, individual=None):
    users = args['users']
    platforms_indexed = args['platforms_indexed']
    minutos_totales_ponderados = 0
    verbose = False

    for i, user in enumerate(users):
        # Creamos una copia del usuario o usamos un objeto nuevo para este individuo
        if individual is None:
            current_individual = type('Individual', (), {})
        else:
            current_individual = individual

        plataformas_por_mes = candidate[i]
        plataformas_mes_dict = {mes: plat_id for mes, plat_id in enumerate(plataformas_por_mes)}
        contenidos_disponibles = []

        # Crear contenidos disponibles con tipo
        for pelicula in user.movies:
            duracion = pelicula['movie_duration']
            if duracion <= 0:
                continue
            meses_disponibles = [
                mes for mes, plat_id in plataformas_mes_dict.items()
                if plat_id in pelicula['platforms']
            ]
            if meses_disponibles:
                contenidos_disponibles.append({
                    'tipo': 'pelicula',
                    'id': pelicula['title'],
                    'duracion': duracion,
                    'interes': pelicula['interest'],
                    'valor_ponderado': duracion * (pelicula['interest']),
                    'meses': meses_disponibles,
                    'eficiencia': pelicula['interest']
                })

        for serie in user.series:
            plataformas_serie = serie.get('platforms', [])
            for temporada in serie['season']:
                duracion = temporada['season_duration']
                if duracion <= 0:
                    continue
                plataformas_temporada = set(temporada.get('platforms', []) + plataformas_serie)
                meses_disponibles = [
                    mes for mes, plat_id in plataformas_mes_dict.items()
                    if plat_id in plataformas_temporada
                ]
                if meses_disponibles:
                    contenidos_disponibles.append({
                        'tipo': 'serie',
                        'id': f"{serie['title']} - SEASON{temporada['season_number']}",
                        'duracion': duracion,
                        'interes': serie['interest'],
                        'valor_ponderado': duracion * (serie['interest']),
                        'meses': meses_disponibles,
                        'eficiencia': serie['interest']
                    })

        # Ordenar por eficiencia
        contenidos_disponibles.sort(key=lambda x: (x['eficiencia'], -x['duracion']), reverse=True)

        minutos_usados_por_mes = {mes: 0 for mes in range(12)}
        contenidos_vistos = set()
        watched_movies = {mes: [] for mes in range(12)}
        watched_series = {mes: [] for mes in range(12)}

        for contenido in contenidos_disponibles:
            meses_ordenados = sorted(contenido['meses'], key=lambda m: minutos_usados_por_mes[m])
            for mes in meses_ordenados:
                clave_contenido = (contenido['id'], mes)
                if clave_contenido not in contenidos_vistos:
                    if minutos_usados_por_mes[mes] + contenido['duracion'] <= user.monthly_minutes:
                        minutos_totales_ponderados += contenido['valor_ponderado']
                        minutos_usados_por_mes[mes] += contenido['duracion']
                        contenidos_vistos.add(clave_contenido)

                        if contenido['tipo'] == 'pelicula':
                            entrada = {
                                "title": contenido['id'],
                                "mes": mes + 1,
                                "plataforma": platforms_indexed.get(
                                    str(plataformas_mes_dict[mes]), f"Plataforma {plataformas_mes_dict[mes]}"
                                ),
                                "duracion": contenido['duracion'],
                                "user_id": f"user_{i}"
                            }
                            watched_movies[mes].append(entrada)
                        else:
                            if ' - T' in contenido['id']:
                                title, season_str = contenido['id'].split(' - SEASON')
                                season_number = int(season_str)
                            else:
                                title = contenido['id']
                                season_number = 1

                            entrada = {
                                "title": title,
                                "season_number": season_number,
                                "mes": mes + 1,
                                "plataforma": platforms_indexed.get(
                                    str(plataformas_mes_dict[mes]), f"Plataforma {plataformas_mes_dict[mes]}"
                                ),
                                "duracion": contenido['duracion'],
                                "user_id": f"user_{i}"
                            }
                            watched_series[mes].append(entrada)

                        if verbose:
                            print(
                                f"  Mes {mes + 1}: {contenido['tipo']} - {contenido['id']} ({contenido['duracion']} min)")

                        break

        monthly_data = []

        for mes in range(12):
            monthly_data.append({
                "month": mes,
                "platform": plataformas_mes_dict[mes],
                "watched_movies": watched_movies[mes],
                "watched_series": watched_series[mes]
            })

        user_id = f"user_{i}"
        if 'monthly_data_by_user' not in args:
            args['monthly_data_by_user'] = {}
        args['monthly_data_by_user'][user_id] = monthly_data


    if verbose:
        print(f"Total minutos ponderados: {minutos_totales_ponderados}")

    # Devolvemos solo el valor de fitness para mantener la compatibilidad
    return minutos_totales_ponderados



def calcular_costo_total(candidate, args):

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

    fitness = []

    for candidate in candidates:
        print(f"Evaluando individuo: {candidate}")
        minutos_ponderados = calcular_minutos_ponderados(candidate, args)
        costo_total = calcular_costo_total(candidate, args)

        fitness.append([costo_total, -minutos_ponderados])

    print(f"✅ Evaluación completada: {len(fitness)} soluciones generadas")
    return fitness



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
