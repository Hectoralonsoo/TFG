"""import requests

# Tu clave de API de TMDb
api_key = 'd566937910f1e5247e09d2f97385dd0a'  # Reemplaza con tu clave de API de TMDb
base_url = 'https://api.themoviedb.org/3'


# Función para obtener las series de un servicio de streaming
def get_series_by_streaming_service(service_id, region='ES'):
    url = f'{base_url}/discover/tv'
    params = {
        'api_key': api_key,
        'with_watch_providers': service_id,  # ID del servicio de streaming
        'watch_region': region,  # Región (por defecto, España)
        'language': 'es-ES',  # Idioma español
        'page': 1  # Página inicial
    }

    all_series = []

    while params['page'] < 10:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            all_series.extend(data['results'])

            # Verificar si hay más páginas de resultados
            if data['page'] < data['total_pages']:
                params['page'] += 1
            else:
                break
        else:
            print(f"Error en la solicitud: {response.status_code}")
            break

    return all_series


# Función para obtener las temporadas de una serie
def get_seasons_of_series(series_id):
    url = f'{base_url}/tv/{series_id}'
    params = {
        'api_key': api_key,
        'language': 'es-ES'  # Idioma español
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data['seasons']
    else:
        print(f"Error al obtener temporadas de la serie {series_id}")
        return []


# ID de Netflix en TMDb
netflix_service_id = 8  # Netflix

# Obtener series de Netflix
series = get_series_by_streaming_service(netflix_service_id)

# Imprimir series y sus temporadas
for serie in series:
    print(f"Serie: {serie['name']}")
    seasons = get_seasons_of_series(serie['id'])
    if seasons:
        for season in seasons:
            print(f"  Temporada {season['season_number']}: {season['name']}")
    else:
        print("  No se encontraron temporadas.")
"""

import requests

api_key = 'd566937910f1e5247e09d2f97385dd0a'  # Reemplaza con tu clave de API de TMDb
base_url = 'https://api.themoviedb.org/3'

def get_seasons(series_id):
    url = f'{base_url}/tv/{series_id}'
    params = {'api_key': api_key, 'language': 'es-ES'}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get('seasons', [])
    return []

def is_available_on_netflix(series_id):
    url = f'{base_url}/tv/{series_id}/watch/providers'
    response = requests.get(url, params={'api_key': api_key})
    if response.status_code == 200:
        providers = response.json().get('results', {}).get('ES', {})
        return any(provider['provider_id'] == 8 for provider in providers.get('flatrate', []))
    return False

def print_netflix_seasons(series_id):
    if is_available_on_netflix(series_id):
        seasons = get_seasons(series_id)
        print("Temporadas de Bob Esponja en Netflix:")
        for season in seasons:
            print(f"  Temporada {season['season_number']}: {season['name']}")
    else:
        print("Bob Esponja no está disponible en Netflix en España.")

# ID de Bob Esponja
series_id = 387
print_netflix_seasons(series_id)
