import json
from pathlib import Path

folder = Path("dataset/labels/aus_energy_bill")

remove_fields = {
    "mirn",
    "gas_mj",
}

for json_file in folder.rglob("*.json"):
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        changed = False
        for field in remove_fields:
            if field in data:
                del data[field]
                changed = True

        if changed:
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"Updated: {json_file}")

    except Exception as e:
        print(f"Error: {json_file} -> {e}")