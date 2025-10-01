import os, csv, pymongo
from pathlib import Path

# לוודא שתיקיית Data קיימת
data_dir = Path("Data")
data_dir.mkdir(exist_ok=True)

# חיבור ל-MongoDB
client = pymongo.MongoClient("mongodb+srv://ADMIN12:rf9lrC4x3PZgbI7M@realestateproject.sqf4qmt.mongodb.net")
col    = client["realestateproject"]["test"]

# נתיב לשמירת הקובץ
output_file = data_dir / "realestate.csv"

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = None
    for doc in col.find({}, {"_id": 0}):  # drop _id if not needed
        if writer is None:                # write header once
            writer = csv.DictWriter(f, fieldnames=doc.keys())
            writer.writeheader()
        writer.writerow(doc)

print(f"✅ File saved to: {output_file.resolve()}")
