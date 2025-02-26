
class User:
    def __init__(self, name, user_id, monthly_minutes, movies, series, months):
        """
        Clase para representar un usuario.

        Args:
            name (str): Nombre del usuario.
            user_id (int): ID del usuario.
            monthly_minutes (int): Minutos disponibles para consumir contenido mensual.
            movies (list): Lista de películas preferidas del usuario.
            series (list): Lista de series favoritas del usuario. Cada serie incluye nombre y temporadas vistas.
        """
        self.name = name
        self.id = user_id
        self.monthly_minutes = monthly_minutes
        self.movies = movies
        self.series = series
        self.months = months


    def __repr__(self):
        return (f"User(name={self.name}, id={self.id}, monthly_minutes={self.monthly_minutes}, "
                f"movies={self.movies}, series={self.series})")

