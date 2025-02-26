import json

from Models.User import User


def load_users_from_json(json_file):

    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        users = [
            User(
                name=user_data["name"],
                user_id=user_data["user_id"],
                monthly_minutes=user_data["monthly_minutes"],
                movies=user_data["movies"],
                series=user_data["series"],
                months=user_data["months"]
            )
            for user_data in data
        ]
    return users

