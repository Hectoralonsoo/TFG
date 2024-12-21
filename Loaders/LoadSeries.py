import json
from Models.Serie import Episode, Season, Serie



def load_series_from_json(json_file):
    try:
        # Abrir y cargar el archivo JSON
        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: El archivo '{json_file}' no existe.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error al leer el JSON: {e}")
        return []

    series_list = []

    for serie_data in data:
        try:
            # Procesar episodios de cada temporada
            seasons = []
            for season_data in serie_data.get("seasons", []):
                episodes = [
                    Episode(
                        episode.get("episode_name", "Sin título"),
                        episode.get("duration", 0)
                    )
                    for episode in season_data.get("episodes", [])
                ]

                season = Season(
                    season_number=season_data.get("season_number", 0),
                    season_name=season_data.get("season_name", "Sin nombre"),
                    season_duration=season_data.get("season_duration", 0),
                    streaming_providers=season_data.get("streaming_providers", []),
                    episodes=episodes
                )
                seasons.append(season)

            # Crear instancia de Serie
            serie = Serie(
                title=serie_data.get("title", "Sin título"),
                streaming_services=serie_data.get("streaming_services", []),
                seasons=seasons
            )
            series_list.append(serie)

        except KeyError as e:
            print(f"Error: Falta la clave {e} en la serie '{serie_data.get('title', 'Desconocida')}'.")

    return series_list
