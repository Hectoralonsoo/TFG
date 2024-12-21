from Loaders import LoadSeries
"""import requests

# Tu clave de API de TMDb
api_key = 'd566937910f1e5247e09d2f97385dd0a'  # Reemplaza con tu clave de API de TMDb
base_url = 'https://api.themoviedb.org/3/discover/movie'

# Parámetros de la solicitud
params = {
    'api_key': api_key,
    'with_watch_providers': '8',  # ID de Netflix en TMDb
    'watch_region': 'ES',  # España
    'language': 'es-ES',  # Para obtener los nombres en español
    'page': 1  # Página inicial
}

# Inicializar una lista para almacenar todas las películas
all_movies = []

# Iterar a través de las páginas para obtener más resultados
while params["page"] < 10:
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        # Agregar los resultados de la página actual a la lista
        all_movies.extend(data['results'])

        # Verificar si hay más páginas de resultados
        if data['page'] < data['total_pages']:
            # Incrementar el número de página
            params['page'] += 1
        else:
            break  # No hay más páginas de resultados, salir del bucle
    else:
        print(f"Error en la solicitud: {response.status_code}")
        break

# Imprimir los nombres de todas las películas obtenidas
for movie in all_movies:
    print(movie['title'])

# Imprimir el número total de películas obtenidas
print(f"\nTotal de películas: {len(all_movies)}")
"""
i = 0
series = LoadSeries.load_series_from_json("Data/series.json")







