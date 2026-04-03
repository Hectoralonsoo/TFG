

class StreamingPlan:
    def __init__(self, service_name, name, profiles, price):
        self.service_name = service_name
        self.name = name
        self.profiles = profiles
        self.price = price

    def __str__(self):
        result = f"Servicio: {self.service_name}\n"
        result += f"    Plan: {self.name}\n"
        result += f"    Perfiles: {self.profiles}\n"
        result += f"    Precio (EUR): {self.price}€\n"
        return result