from . import Content



class Episode:
    def __init__(self, episode_name, duration):
        self.episode_name = episode_name
        self.duration = duration

    def __str__(self):
        return f"Episode: {self.episode_name}, Duration: {self.duration} minutes"


class Season:
    def __init__(self, season_number, season_name, season_duration, streaming_providers, episodes):
        self.season_number = season_number
        self.season_name = season_name
        self.season_duration = season_duration
        self.streaming_providers = streaming_providers
        self.episodes = episodes  # List of Episode objects

    def __str__(self):
        episodes_info = "\n    ".join(str(episode) for episode in self.episodes)
        return (f"Season {self.season_number} - {self.season_name}, Duration: {self.season_duration} minutes\n"
                f"  Streaming Providers: {', '.join(self.streaming_providers)}\n"
                f"  Episodes:\n    {episodes_info}")


class Serie(Content.content):
    def __init__(self, title, seasons):
        super().__init__(title)
        self.seasons = seasons  # List of Season objects

    def __str__(self):
        seasons_info = "\n".join(str(season) for season in self.seasons)
        return (f"Series: {self.title}\n"
                f"  Available on: {', '.join(self.streaming_services)}\n"
                f"  Seasons:\n{seasons_info}")

