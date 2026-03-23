import json
import random
from pathlib import Path
from src.tfg.models.User import User
from src.tfg.paths import DATA_DIR

# Load existing data
def load_json_data(filename):
    with open(DATA_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)

movies = load_json_data("movies.json")
series = load_json_data("series.json")
streaming_plans = load_json_data("streamingPlans.json")



platforms = list(streaming_plans.keys())




# Helper functions
def generate_user(user_id, max_movies=250, max_series=250):
    # Filtrar películas y series que tengan al menos una plataforma disponible
    available_movies = [m for m in movies if m["platforms"]]
    available_series = [s for s in series if s["platforms"]]

    # Seleccionar aleatoriamente solo de las disponibles
    selected_movies = random.sample(available_movies, k=random.randint(1, min(max_movies, len(available_movies))))
    selected_series = random.sample(available_series, k=random.randint(1, min(max_series, len(available_series))))

    # Calcular minutos de visualización mensual (base ±50%)
    base_minutes = 2000
    monthly_minutes = int(base_minutes * (0.5 + random.random()))  # Entre 900 y 2700 minutos

    # Crear usuario
    user = User(
        name=f"User_{user_id}",
        user_id=user_id,
        monthly_minutes=monthly_minutes,
        movies=[],
        series=[],
        months = [],
        watched_movies=[],
        watched_series=[]
    )
    print(f"🧑‍💻 Usuario generado: {user.__dict__}")

    for movie in selected_movies:
        interest = round(random.uniform(0.5, 1.0), 2)
        user.movies.append({
            "title": movie["title"],
            "movie_duration":movie["duration"],
            "platforms": movie["platforms"],
            "interest": interest
        })


    for serie in selected_series:
        interest = round(random.uniform(0.5, 1.0), 2)
        available_seasons = []
        for season in serie["seasons"]:
            if season.get("platforms"):  # Se ignoran temporadas sin plataformas
                season_copy = {k: v for k, v in season.items() if k != "episodes"}
                available_seasons.append(season_copy)

        available_seasons.sort(key=lambda s: s.get("season_number", 0))

        if available_seasons:
            start_index = random.randint(0, len(available_seasons) - 1)
            selected_seasons = available_seasons[start_index:]
        else:
            selected_seasons = []

        user.series.append({
            "title": serie["title"],
            "season": selected_seasons,
            "season_duration":season ["season_duration"],
            "platforms": serie["platforms"],
            "interest": interest
        })

    return user


users = [generate_user(user_id) for user_id in range(1, 201)]


def save_users_to_json(users, filename):
    users_data = []
    for user in users:
        user_data = {
            "name": user.name,
            "user_id": user.id,
            "monthly_minutes": user.monthly_minutes,
            "movies": user.movies,
            "series": user.series,
            "months": user.months,
            "watched_movies": user.watched_movies,
            "watched_series": user.watched_series
        }
        users_data.append(user_data)

    with open(DATA_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(users_data, f, indent=4, ensure_ascii=False)

def update_users_json(users_data, watched_movies=None, watched_series=None):
    """
    Actualiza el archivo `users.json` con las películas, series o historial visto por usuario.
    Soporta objetos tipo `User`.
    """
    print(f"🔄 Intentando actualizar `users.json` con {len(users_data)} usuarios...")

    if not users_data:
        print("⚠️ ERROR: No hay usuarios en `users_data`. Verificar generación de usuarios.")
        return

    updated_users = []

    for user in users_data:
        # Convertimos el objeto User a diccionario
        user_dict = {
            "name": user.name,
            "user_id": user.id,
            "monthly_minutes": user.monthly_minutes,
            "movies": user.movies,
            "series": user.series,
            "months": user.months,
            "watched_movies": user.watched_movies,
            "watched_series": user.watched_series,
            "historial": user.historial if hasattr(user, "historial") else {}
        }

        updated_users.append(user_dict)

    # Guardar el archivo actualizado
    output_path = DATA_DIR / "users.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(updated_users, file, ensure_ascii=False, indent=4)

    print(f"✅ `users.json` actualizado correctamente en {output_path}.")




save_users_to_json(users, "users5.json")
print("✅ Users generated successfully at:", DATA_DIR / "users5.json")
