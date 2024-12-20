import requests
import json

# Configuración
api_key = "d566937910f1e5247e09d2f97385dd0a"
tv_id = "1399"  # Ejemplo: Game of Thrones

# URL base
base_url = "https://api.themoviedb.org/3"

def getSeasonStreamingProviders(serie_id, season_number, country_code='ES'):
    url = f"{base_url}/tv/{serie_id}/season/{season_number}/watch/providers?api_key={api_key}"
    response = requests.get(url).json()
    providers = response.get("results", {}).get(country_code, {}).get("flatrate", [])
    return [provider["provider_name"] for provider in providers]

def getBestMovies(cantidad):
    peliculas = []
    pagina = 1
    while len(peliculas) < cantidad:
        url = f"{base_url}/movie/top_rated?api_key={api_key}&language=en-US&page={pagina}"
        response = requests.get(url).json()
        peliculas.extend(response.get("results", []))
        pagina += 1
    return peliculas[:cantidad]

def getBestSeries(cantidad):
    series = []
    page = 1
    while len(series) < cantidad:
        url = f"{base_url}/tv/top_rated?api_key={api_key}&language=en-US&page={page}"
        response = requests.get(url).json()
        series.extend(response.get("results", []))
        page += 1
    return series[:cantidad]

def saveJson(nombre_archivo, data):
    with open(nombre_archivo, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)



def getMovieDuration(movie_id):
    url = f"{base_url}/movie/{movie_id}?api_key={api_key}&language=en-US"
    movie_data = requests.get(url).json()
    return movie_data.get("runtime", "Desconocido")

def searchSeriesByName(name):
    url = f"{base_url}/search/tv?api_key={api_key}&query={name}&language=es-ES"
    response = requests.get(url).json()
    series = response.get("results", [])
    if series:
        return series[0]  # Retorna la primera serie que coincida
    return None



def getEpisodeDuration(serie_id, season_number):
    url = f"{base_url}/tv/{serie_id}/season/{season_number}?api_key={api_key}&language=en-US"
    response = requests.get(url).json()
    episodes = response.get("episodes", [])

    episodes_duration = []
    total_duration = 0  # Variable para sumar las duraciones de los episodios

    for episode in episodes:
        episode_name = episode.get("name", "Desconocido")
        episode_duration = episode.get("runtime", 0) or 0  # Si no tiene duración, consideramos 0
        total_duration += episode_duration
        episodes_duration.append({
            "episode_name": episode_name,
            "duration": episode_duration
        })

    return total_duration, episodes_duration


def getStreamingProviders(media_type, media_id, country_code):
    url = f"{base_url}/{media_type}/{media_id}/watch/providers?api_key={api_key}"
    response = requests.get(url).json()
    providers = response.get("results", {}).get(country_code, {}).get("flatrate", [])
    return [provider["provider_name"] for provider in providers]

# Obtener la duración total de las temporadas de una serie
def getSeasonDuration(serie_id):
    url = f"{base_url}/tv/{serie_id}?api_key={api_key}&language=en-US"
    serie_data = requests.get(url).json()

    seasons_duration = []

    for season in serie_data.get("seasons", []):
        season_number = season["season_number"]
        season_name = season["name"]

        # Obtener la duración total de los episodios y la lista de episodios
        season_duration, episodes_duration = getEpisodeDuration(serie_id, season_number)
        season_streaming_providers = getSeasonStreamingProviders(serie_id, season_number)

        seasons_duration.append({
            "season_number": season_number,
            "season_name": season_name,
            "season_duration": season_duration,
            "streaming_providers": season_streaming_providers,
            "episodes": episodes_duration

        })

    return seasons_duration


def getAndSaveSeries(cantidad, archivo_salida):
    mejores_series = getBestSeries(cantidad)
    series_duracion = []
    for serie in mejores_series:
        duraciones_temporadas = getSeasonDuration(serie["id"])
        streaming_services = getStreamingProviders("tv", serie["id"], "ES")
        series_duracion.append({
            "title": serie["name"],
            "streaming_services": streaming_services,
            "seasons": duraciones_temporadas,
        })
    saveJson(archivo_salida, series_duracion)
    print(f"Datos guardados en {archivo_salida}.")


def getAndSaveMovie(cantidad, archivo_salida):
    bestMovies = getBestMovies(cantidad)
    movieDuration = []
    for movie in bestMovies:
        duracion = getMovieDuration(movie["id"])
        streaming_services = getStreamingProviders("movie", movie["id"], "ES")
        movieDuration.append({
            "title": movie["title"],
            "duration": duracion,
            "streaming_services": streaming_services
        })
    saveJson(archivo_salida, movieDuration)
    print(f"Datos guardados en {archivo_salida}.")

getAndSaveMovie(30, "movies.json")
getAndSaveSeries(10, "series.json")

serie_name = "SpongeBob SquarePants"
serie_data = searchSeriesByName(serie_name)

if serie_data:
    serie_id = serie_data["id"]
    print(f"Información de la serie: {serie_data['name']}")

    # Obtener la duración de las temporadas y los proveedores de streaming
    duraciones_temporadas = getSeasonDuration(serie_id)

    # Obtener los proveedores de streaming de la serie en general
    streaming_services = getStreamingProviders("tv", serie_id, "ES")

    # Crear un diccionario con la información de la serie
    serie_info = {
        "title": serie_data["name"],
        "overview": serie_data.get("overview", "Sin descripción"),
        "streaming_services": streaming_services,
        "seasons": duraciones_temporadas,
    }

    # Guardar la información en un archivo JSON
    saveJson("bob_esponja_info.json", serie_info)
    print(f"Datos de {serie_name} guardados en 'bob_esponja_info.json'.")
else:
    print("No se encontró la serie.")