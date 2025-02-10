import json

# 📁 Archivos de entrada y salida
INPUT_FILE = "movies.json"
OUTPUT_FILE = "MoviesPlatform.json"


def process_movies_data():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        movies_data = json.load(f)

    platform_dict = {}

    for movie in movies_data:
        movie_title = movie["title"]
        movie_duration = movie["duration"]
        movie_platforms = movie["platforms"]  # 🔹 Tomamos directamente las plataformas de cada película

        for platform in movie_platforms:
            if platform not in platform_dict:
                platform_dict[platform] = {}  # Crea la plataforma si no existe

            if movie_title not in platform_dict[platform]:
                platform_dict[platform][movie_title] = movie_duration  # Agrega la película con su duración

    # 📂 Guarda el JSON final
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(platform_dict, f, indent=4, ensure_ascii=False)

    print(f"✅ Archivo guardado: {OUTPUT_FILE}")


process_movies_data()
