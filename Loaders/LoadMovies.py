import json
from Models.Movie import Movie

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

file = "../Data/movies.json"

movie_list = load_movies_from_json(file)
for movie in movie_list:
    print(movie)
