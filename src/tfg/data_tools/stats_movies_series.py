import json
import statistics


def calcular():


    try:
        with open('movies.json', 'r', encoding='utf-8') as f:
            movies = json.load(f)
        m_count = len(movies)
        m_durations = [m['duration'] for m in movies if isinstance(m['duration'], (int, float))]
        m_platforms = [len(m['platforms']) for m in movies]
        print("-" * 30)
        print("ESTADÍSTICAS PELÍCULAS")
        print(f"Total: {m_count}")
        print(f"Duración Media: {statistics.mean(m_durations):.2f} min (SD: {statistics.stdev(m_durations):.2f})")
        print(f"Plataformas Media: {statistics.mean(m_platforms):.2f} (SD: {statistics.stdev(m_platforms):.2f})")
    except Exception as e:
        print(f"Error con movies.json: {e}")

    try:
        with open('series.json', 'r', encoding='utf-8') as f:
            series = json.load(f)
        s_count = len(series)
        s_platforms = [len(s['platforms']) for s in series]
        s_seasons_count = [len(s['seasons']) for s in series]
        s_durations = []
        for s in series:
            total_duration = sum(season['season_duration'] for season in s['seasons'])
            s_durations.append(total_duration)

        print("-" * 30)
        print("ESTADÍSTICAS SERIES")
        print(f"Total: {s_count}")
        print(f"Duración Total Media: {statistics.mean(s_durations):.2f} min (SD: {statistics.stdev(s_durations):.2f})")
        print(f"Plataformas Media: {statistics.mean(s_platforms):.2f} (SD: {statistics.stdev(s_platforms):.2f})")
        print(f"Temporadas Media: {statistics.mean(s_seasons_count):.2f} (SD: {statistics.stdev(s_seasons_count):.2f})")
        print("-" * 30)

    except Exception as e:
        print(f"Error con series.json: {e}")

if __name__ == "__main__":
    calcular()