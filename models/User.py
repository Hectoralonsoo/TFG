
class User:
    def __init__(self, name, user_id, monthly_minutes, movies=None, series=None, months=None,
                 watched_movies=None, watched_series=None):
        """
        Clase para representar un usuario.
        """
        self.name = name
        self.id = user_id
        self.monthly_minutes = monthly_minutes
        self.movies = movies if movies is not None else []
        self.series = series if series is not None else []
        self.months = months if months is not None else []
        self.watched_movies = watched_movies
        self.watched_series = watched_series
        self.historial = {}

    def __repr__(self):
        return (f"User(name={self.name}, id={self.id}, monthly_minutes={self.monthly_minutes}, "
                f"movies={len(self.movies)} pelis, series={len(self.series)} series)")


