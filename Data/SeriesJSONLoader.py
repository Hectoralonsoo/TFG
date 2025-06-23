import json

INPUT_FILE = "series.json"
OUTPUT_FILE = "SeriesPlatform.json"


def process_series_data():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        series_data = json.load(f)

    platform_dict = {}

    for series in series_data:
        series_title = series["title"]

        for season in series["seasons"]:
            season_name = season["season_name"]
            season_duration = sum(episode["duration"] for episode in season["episodes"])
            season_platforms = season["platforms"]

            for platform in season_platforms:
                if platform not in platform_dict:
                    platform_dict[platform] = {}

                if series_title not in platform_dict[platform]:
                    platform_dict[platform][series_title] = {}  # Crea la serie si no existe

                platform_dict[platform][series_title][season_name] = season_duration  # Agrega la temporada con su duración


    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(platform_dict, f, indent=4, ensure_ascii=False)

    print(f"✅ Archivo guardado: {OUTPUT_FILE}")


process_series_data()

