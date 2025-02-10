import json
import random
from pathlib import Path
from Models.User import User

# Paths configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# Load existing data
def load_json_data(filename):
    with open(DATA_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)


movies = load_json_data("movies.json")
series = load_json_data("series.json")
streaming_plans = load_json_data("streamingPlans.json")

# Extract platforms from streaming_plans (assuming streaming_plans is a list of dicts)
platforms = [list(platform.keys())[0] for platform in streaming_plans]


# Helper functions
def generate_user(user_id, max_movies=10, max_series=10):
    # Filtrar películas y series que tengan al menos una plataforma disponible
    available_movies = [m for m in movies if m["platforms"]]
    available_series = [s for s in series if s["platforms"]]

    # Seleccionar aleatoriamente solo de las disponibles
    selected_movies = random.sample(available_movies, k=random.randint(1, min(max_movies, len(available_movies))))
    selected_series = random.sample(available_series, k=random.randint(0, min(max_series, len(available_series))))

    # Calcular minutos de visualización mensual (base ±50%)
    base_minutes = 1800
    monthly_minutes = int(base_minutes * (0.5 + random.random()))  # 900-2700 mins

    # Crear usuario
    user = User(
        name=f"User_{user_id}",
        user_id=user_id,
        monthly_minutes=monthly_minutes,
        movies=[],
        series=[]
    )

    # Agregar preferencias con interés y plataformas disponibles
    for movie in selected_movies:
        interest = round(random.uniform(0.5, 1.0), 2)
        user.movies.append({
            "title": movie["title"],
            "platforms": movie["platforms"],
            "interest": interest
        })


    for serie in selected_series:
        interest = round(random.uniform(0.5, 1.0), 2)
        user.series.append({
            "title": serie["title"],
            "season":serie["seasons"],
            "platforms": serie["platforms"],
            "interest": interest
        })


    return user




# Generate 100 users
users = [generate_user(user_id) for user_id in range(1, 101)]


# Save to JSON
def save_users_to_json(users, filename):
    users_data = []
    for user in users:
        user_data = {
            "name": user.name,
            "user_id": user.id,
            "monthly_minutes": user.monthly_minutes,
            "movies": user.movies,
            "series": user.series,
        }
        users_data.append(user_data)

    with open(DATA_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(users_data, f, indent=4, ensure_ascii=False)


save_users_to_json(users, "users.json")
print("✅ Users generated successfully at:", DATA_DIR / "users.json")