import json

def load_platforms_json(json_file):
    try:
        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)
            print("Datos cargados correctamente:")
            return data  # ✅ Devuelve el dict directamente
    except FileNotFoundError:
        print(f"Error: El archivo '{json_file}' no existe.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error al leer el JSON: {e}")
        return {}
