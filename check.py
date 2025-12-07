# check_mongo_data.py

from pymongo import MongoClient
from config import Config

COLLECTION_NAME = "player_season_predictions"

def main():
    client = MongoClient(Config.MONGO_URI)
    db = client[Config.MONGO_DB_NAME]
    coll = db[COLLECTION_NAME]

    print("🔎 One example document:")
    doc = coll.find_one()
    print(doc)

    print("\n🔎 Example: CSK 2024 players:")
    for d in coll.find({"team": "Chennai Super Kings", "season": "2024"}, {"_id": 0, "player_name": 1, "predicted_points": 1}):
        print(d)

if __name__ == "__main__":
    main()
