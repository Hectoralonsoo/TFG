from . import Content

class Movie(Content.content):
    def __init__(self, title, streaming_services, duration):
        super().__init__(title)
        self.streaming_services = streaming_services
        self.duration = duration

    def __str__(self):
        services = ', '.join(self.streaming_services) if self.streaming_services else "No disponible"
        return f"Title: {self.title}\n Duration: {self.duration} minutos\nServicios de streaming: {services}"

