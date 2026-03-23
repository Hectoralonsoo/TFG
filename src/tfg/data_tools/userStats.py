import json
import statistics
import os

files = ['users1.json', 'users2.json', 'users3.json', 'users4.json', 'users5.json']

# Cabecera de la tabla que saldrá por consola
print(
    f"{'Dataset':<12} | {'Users':<5} | {'Mov/User':<8} | {'Ser/User':<8} | {'Seas/User':<9} | {'Min/Mes':<8} | {'Interest':<8}")
print("-" * 90)

for filename in files:
    try:
        if not os.path.exists(filename):
            print(f"{filename:<12} | NO ENCONTRADO")
            continue
        with open(filename, 'r', encoding='utf-8') as f:
            users = json.load(f)
        num_users = len(users)
        # Listas para guardar los conteos de cada usuario
        movies_per_user = []
        series_per_user = []
        seasons_per_user = []
        minutes_per_user = []
        all_interests = []  # Bolsa común de intereses
        for u in users:
            # 1. Películas
            m_list = u.get('movies', [])
            movies_per_user.append(len(m_list))
            # 2. Series y Temporadas
            s_list = u.get('series', [])
            series_per_user.append(len(s_list))
            # Contar temporadas totales de este usuario (sumando las de cada serie)
            # Nota: Tu JSON usa la clave "season" (singular) para la lista de temporadas
            total_seasons = sum(len(s.get('season', [])) for s in s_list)
            seasons_per_user.append(total_seasons)
            # 3. Minutos mensuales
            minutes_per_user.append(u.get('monthly_minutes', 0))
            # 4. Recopilar intereses (de pelis y series)
            for m in m_list:
                if 'interest' in m: all_interests.append(m['interest'])
            for s in s_list:
                if 'interest' in s: all_interests.append(s['interest'])
        # Cálculos de medias
        avg_mov = statistics.mean(movies_per_user) if movies_per_user else 0
        avg_ser = statistics.mean(series_per_user) if series_per_user else 0
        avg_sea = statistics.mean(seasons_per_user) if seasons_per_user else 0
        avg_min = statistics.mean(minutes_per_user) if minutes_per_user else 0
        avg_int = statistics.mean(all_interests) if all_interests else 0
        # Imprimir fila
        print(
            f"{filename:<12} | {num_users:<5} | {avg_mov:<8.1f} | {avg_ser:<8.1f} | {avg_sea:<9.1f} | {avg_min:<8.0f} | {avg_int:<8.2f}")

    except Exception as e:
        print(f"Error leyendo {filename}: {e}")