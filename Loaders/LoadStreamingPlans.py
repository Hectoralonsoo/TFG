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

    for service_name, service_data in data.items():
        try:
            # Verificar si el servicio tiene un solo plan o varios
            plans = service_data.get("planes", [])
            if not plans:  # Si no tiene planes, puede que tenga solo un plan
                plan = service_data.get("plan")
                if plan:
                    plans = [{"plan": plan, "precio": service_data.get("precio", 0),
                              "perfiles": service_data.get("perfiles", 0)}]

            for plan in plans:
                # Crear instancia de StreamingPlan
                streaming_plan = StreamingPlan(
                    service_name=service_name,
                    name=plan.get("plan", "Sin nombre"),
                    profiles=plan.get("perfiles", 0),
                    price=plan.get("precio", 0)
                )
                streaming_plans_list.append(streaming_plan)
        except Exception as e:
            print(f"Error al procesar el servicio '{service_name}': {e}")

    return streaming_plans_list


def getServices(streaming_plan_list):
    service_and_plans = [f"{plan.service_name} {plan.name}" for plan in streaming_plan_list]
    return service_and_plans
