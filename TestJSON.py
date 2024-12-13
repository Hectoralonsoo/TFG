import json

with open('streamingPlans.json', 'r', encoding='utf-8') as file:
    streaming_data = json.load(file)

service_names = [service_name for service in streaming_data for service_name in service.keys()]
number_of_services = len(service_names)
print(service_names)
print("Número total de servicios:", number_of_services)