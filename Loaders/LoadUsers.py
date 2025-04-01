import json

from Models.User import User


def load_users_from_json(json_file):
    users = []

    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

        for user_data in data:
            try:
                user = User(
                    name=user_data["name"],
                    user_id=user_data["user_id"],
                    monthly_minutes=user_data["monthly_minutes"],
                    movies=user_data.get("movies", []),
                    series=user_data.get("series", []),
                    months=user_data.get("months", []),
                    watched_movies=user_data.get("watched_movies", []),
                    watched_series=user_data.get("watched_series", [])
                )
                users.append(user)
            except Exception as e:
                print(f"❌ Error cargando usuario: {user_data.get('name', 'desconocido')} → {e}")

    print(f"✅ Usuarios cargados: {len(users)}")
    return users


