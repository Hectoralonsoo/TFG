import requests
import json

api_key = "d566937910f1e5247e09d2f97385dd0a"

base_url = "https://api.themoviedb.org/3"

with open("indice_plataformas.json", "r", encoding="utf-8") as file:
    platform_name_to_id = json.load(file)

def convert_platforms_to_ids(platform_names):
    return [int(id) for id, name in platform_name_to_id.items() if name in platform_names]

def getSeasonStreamingProviders(serie_id, season_number, country_code='ES'):
    url = f"{base_url}/tv/{serie_id}/season/{season_number}/watch/providers?api_key={api_key}"
    response = requests.get(url).json()
    providers = response.get("results", {}).get(country_code, {}).get("flatrate", [])
    platform_names =  [provider["provider_name"] for provider in providers]
    return convert_platforms_to_ids(platform_names)

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


def getEpisodeDuration(serie_id, season_number):
    url = f"{base_url}/tv/{serie_id}/season/{season_number}?api_key={api_key}&language=en-US"
    response = requests.get(url).json()
    episodes = response.get("episodes", [])

    episodes_duration = []
    total_duration = 0

    for episode in episodes:
        episode_name = episode.get("name", "Desconocido")
        episode_duration = episode.get("runtime", 0) or 0
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
    platform_names = [provider["provider_name"] for provider in providers]
    print(platform_names)
    return convert_platforms_to_ids(platform_names)

def getSeasonDuration(serie_id):
    url = f"{base_url}/tv/{serie_id}?api_key={api_key}&language=en-US"
    serie_data = requests.get(url).json()

    seasons_duration = []

    for season in serie_data.get("seasons", []):
        season_number = season["season_number"]
        season_name = season["name"]

        season_duration, episodes_duration = getEpisodeDuration(serie_id, season_number)
        season_streaming_providers = getSeasonStreamingProviders(serie_id, season_number)
        if season_streaming_providers:
            seasons_duration.append({
                "season_number": season_number,
                "season_name": season_name,
                "season_duration": season_duration,
                "platforms": season_streaming_providers,
                "episodes": episodes_duration

            })

    return seasons_duration


def getAndSaveSeries(cantidad, archivo_salida):
    mejores_series = getBestSeries(cantidad)
    i = 0
    series_duracion = []
    for serie in mejores_series:
        print(i)
        duraciones_temporadas = getSeasonDuration(serie["id"])
        i = i+1
        streaming_services = getStreamingProviders("tv", serie["id"], "ES")
        if streaming_services:
            series_duracion.append({
                "title": serie["name"],
                "platforms": streaming_services,
                "seasons": duraciones_temporadas,
            })
    saveJson(archivo_salida, series_duracion)
    print(f"Datos guardados en {archivo_salida}.")


def getAndSaveMovie(cantidad, archivo_salida):
    bestMovies = getBestMovies(cantidad)
    i = 0
    movieDuration = []
    for movie in bestMovies:
        print(i)
        duracion = getMovieDuration(movie["id"])
        i = i+1
        streaming_services = getStreamingProviders("movie", movie["id"], "ES")
        if streaming_services:
            movieDuration.append({
                "title": movie["title"],
                "duration": duracion,
                "platforms": streaming_services
            })
    saveJson(archivo_salida, movieDuration)
    print(f"Datos guardados en {archivo_salida}.")

getAndSaveMovie(2000, "movies.json")
getAndSaveSeries(2000, "series.json")