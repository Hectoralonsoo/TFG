import json



class Content:
    def __init__(self, title):
        self.title = title

    def __str__(self):
        return f"Content: {self.title}"

class Movie(Content):
    def __init__(self, title, streaming_services, duration):
        super().__init__(title)
        self.streaming_services = streaming_services
        self.duration = duration

    def __str__(self):
        services = ', '.join(self.streaming_services) if self.streaming_services else "No disponible"
        return f"Title: {self.title}\n Duration: {self.duration} minutos\nServicios de streaming: {services}"



class Episode:
    def __init__(self, episode_name, duration):
        self.episode_name = episode_name
        self.duration = duration

    def __str__(self):
        return f"Episode: {self.episode_name}, Duration: {self.duration} minutes"


class Season:
    def __init__(self, season_number, season_name, season_duration, streaming_providers, episodes):
        self.season_number = season_number
        self.season_name = season_name
        self.season_duration = season_duration
        self.streaming_providers = streaming_providers
        self.episodes = episodes  # List of Episode objects

    def __str__(self):
        episodes_info = "\n    ".join(str(episode) for episode in self.episodes)
        return (f"Season {self.season_number} - {self.season_name}, Duration: {self.season_duration} minutes\n"
                f"  Streaming Providers: {', '.join(self.streaming_providers)}\n"
                f"  Episodes:\n    {episodes_info}")


class Serie(Content):
    def __init__(self, title, streaming_services, seasons):
        super().__init__(title)
        self.streaming_services = streaming_services
        self.seasons = seasons  # List of Season objects

    def __str__(self):
        seasons_info = "\n".join(str(season) for season in self.seasons)
        return (f"Series: {self.title}\n"
                f"  Available on: {', '.join(self.streaming_services)}\n"
                f"  Seasons:\n{seasons_info}")

class StreamingPlan:
    def __init__(self, service_name, name, profiles, price):
        self.service_name = service_name
        self.name = name
        self.profiles = profiles
        self.price = price

    def __str__(self):
        result = f"Servicio: {self.service_name}\n"
        result += f"    Plan: {self.name}\n"
        result += f"    Perfiles: {self.profiles}\n"
        result += f"    Precio (EUR): {self.price}€\n"
        return result

def load_streaming_plan_json(json_file):
    try:
        # Abrir y cargar el archivo JSON
        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)
            print("Datos cargados correctamente:")  # Verificamos si los datos se cargaron
            print(data)
    except FileNotFoundError:
        print(f"Error: El archivo '{json_file}' no existe.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error al leer el JSON: {e}")
        return []

    streaming_plans_list = []


    for service_name, service_data in data[0].items():
        try:
            plans = service_data.get("plans", [])
            for plan in plans:
                # Crear instancia de StreamingPlan
                streaming_plan = StreamingPlan(
                    service_name=service_name,
                    name=plan.get("name", "Sin nombre"),
                    profiles=plan.get("profiles", 0),
                    price=plan.get("price_eur", 0)
                )
                streaming_plans_list.append(streaming_plan)
        except Exception as e:
            print(f"Error al procesar el servicio '{service_name}': {e}")

    return streaming_plans_list

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


def load_movies_from_json(json_file):
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

    movies_list = []

    # Procesar cada película en los datos
    for movie_data in data:
        try:
            # Crear instancia de Movie
            movie = Movie(
                title=movie_data.get("title", "Sin título"),
                duration=movie_data.get("duration", 0),
                streaming_services=movie_data.get("streaming_services", [])
            )
            movies_list.append(movie)
        except Exception as e:
            print(f"Error al procesar la película: {e}. Datos: {movie_data}")

    return movies_list

    # Nombre del archivo JSON
json_series_file = "series.json"
json_movies_file = "movies.json"
json_streaming_plans_file = "streamingPlans.json"
    # Cargar series desde el JSON

try:
   series_list = load_series_from_json(json_series_file)
   for serie in series_list:
      print(serie)
except FileNotFoundError:
      print(f"Error: El archivo {json_series_file} no se encontró.")
except json.JSONDecodeError:
      print(f"Error: El archivo {json_series_file} no es un JSON válido.")


try:
   movies_list = load_movies_from_json(json_movies_file)
   for movie in movies_list:
      print(movie)
except FileNotFoundError:
      print(f"Error: El archivo {json_series_file} no se encontró.")
except json.JSONDecodeError:
      print(f"Error: El archivo {json_series_file} no es un JSON válido.")

try:
   stream_plan_list = load_streaming_plan_json(json_streaming_plans_file)
   for stream_plan in stream_plan_list:
      print(stream_plan)
except FileNotFoundError:
      print(f"Error: El archivo {json_streaming_plans_file} no se encontró.")
except json.JSONDecodeError:
      print(f"Error: El archivo {json_streaming_plans_file} no es un JSON válido.")



print(stream_plan_list[1])
print(movies_list[5])
print(series_list[0])