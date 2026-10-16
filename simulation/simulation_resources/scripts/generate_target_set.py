import json
import os
import random
from shapely.geometry import Point, Polygon

# Define the boundary polygon from the given coordinates
vertices = [
    (47, -24),
    (34,  27),
    (-31, 12),
    (-19, -39),
]

polygon = Polygon(vertices)

# Generate 30 random points inside the polygon
NUM_OBJECTS = 30
objects = []

min_x, min_y, max_x, max_y = polygon.bounds

while len(objects) < NUM_OBJECTS:
    x = random.uniform(min_x, max_x)
    y = random.uniform(min_y, max_y)
    point = Point(x, y)
    if polygon.contains(point):
        objects.append({
            "id": len(objects) + 1,
            "x": round(x, 4),
            "y": round(y, 4),
            "z": 0
        })

output = {"targets": objects}

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "custom_objects", "target_config.json")
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"Generated {NUM_OBJECTS} positions inside the polygon.")
print(json.dumps(output, indent=2))