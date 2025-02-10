import json
from Models.StreamingPlan import StreamingPlan

def load_streaming_plan_json(json_file):
    try:
        # Abrir y cargar el archivo JSON
        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)
            print("Datos cargados correctamente:")  # Verificamos si los datos se cargaron
            print(data)
    except FileNotFoundError:
        print(f"Error: El archivo '{json_file}' no existe.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error al leer el JSON: {e}")
        return []

    streaming_plans_list = []


    for service_name, service_data in data[0].items():
        try:
            plans = service_data.get("plans", [])
            for plan in plans:
                # Crear instancia de StreamingPlan
                streaming_plan = StreamingPlan(
                    service_name=service_name,
                    name=plan.get("name", "Sin nombre"),
                    profiles=plan.get("profiles", 0),
                    price=plan.get("price_eur", 0)
                )
                streaming_plans_list.append(streaming_plan)
        except Exception as e:
            print(f"Error al procesar el servicio '{service_name}': {e}")

    return streaming_plans_list
"""
file = "../Data/streamingPlans.json"
streaming_plan_list = load_streaming_plan_json(file)
"""
def getServices(streaming_plan_list):
    service_and_plans = [f"{plan.service_name} {plan.name}" for plan in streaming_plan_list]
    return service_and_plans

"""
for plan in streaming_plan_list:
    print(plan)
    print(getServices(streaming_plan_list))
"""