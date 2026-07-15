#!/usr/bin/env python3
"""Generate a CUSTOM_OBJECTS config for ArUco markers."""

import argparse
import json
import random
from pathlib import Path


def make_marker(marker_id, model, x, y, z):
    return {
        "id": marker_id,
        "name": f"aruco_{marker_id}",
        "model": model,
        "x": x,
        "y": y,
        "z": z,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "static": True,
    }


def build_under_drone_objects(args):
    objects = []
    marker_id = 0

    for i in range(args.num_quads):
        objects.append(make_marker(marker_id, args.model, args.x_offset + i * args.spacing, args.y_offset - i * args.spacing, args.z))
        marker_id += 1

    for i in range(args.num_vtols):
        objects.append(make_marker(marker_id, args.model, args.x_offset + i * args.spacing, args.y_offset + 2.0 + i * args.spacing, args.z))
        marker_id += 1

    return objects


def build_random_objects(args):
    rng = random.Random(args.seed)
    objects = []
    attempts = 0
    max_attempts = max(args.count * 100, 100)

    while len(objects) < args.count and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(args.x_min, args.x_max)
        y = rng.uniform(args.y_min, args.y_max)

        if args.min_distance > 0.0:
            too_close = any(((x - obj["x"]) ** 2 + (y - obj["y"]) ** 2) ** 0.5 < args.min_distance for obj in objects)
            if too_close:
                continue

        marker_id = len(objects)
        objects.append(make_marker(marker_id, args.model, round(x, 3), round(y, 3), args.z))

    if len(objects) != args.count:
        raise RuntimeError(
            f"Could only place {len(objects)} of {args.count} markers. Reduce --count/--min-distance or expand bounds."
        )

    return objects


def main():
    parser = argparse.ArgumentParser(
        description="Generate simulation_resources/patches/custom_objects_config.json for ArUco markers."
    )
    parser.add_argument("--placement", choices=("under-drones", "random"), default="under-drones")
    parser.add_argument("--num-quads", type=int, required=True)
    parser.add_argument("--num-vtols", type=int, required=True)
    parser.add_argument("--x-offset", type=float, default=0.0)
    parser.add_argument("--y-offset", type=float, default=0.0)
    parser.add_argument(
        "--z",
        type=float,
        default=0.03,
        help="World Z coordinate for the marker. For terrain worlds, set this to the local ground height.",
    )
    parser.add_argument("--spacing", type=float, default=2.0)
    parser.add_argument("--model", default="aruco_5x5_0")
    parser.add_argument("--count", type=int, default=None, help="Number of random markers. Defaults to num-quads + num-vtols.")
    parser.add_argument("--x-min", type=float, default=-20.0)
    parser.add_argument("--x-max", type=float, default=20.0)
    parser.add_argument("--y-min", type=float, default=-20.0)
    parser.add_argument("--y-max", type=float, default=20.0)
    parser.add_argument("--min-distance", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output",
        default="/aas/simulation_resources/patches/custom_objects_config.json",
    )
    args = parser.parse_args()
    if args.count is None:
        args.count = args.num_quads + args.num_vtols

    if args.placement == "random" and (args.x_min > args.x_max or args.y_min > args.y_max):
        raise ValueError("Random bounds are invalid: min must be lower than max.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    objects = build_random_objects(args) if args.placement == "random" else build_under_drone_objects(args)
    data = {"objects": objects}
    output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(data['objects'])} ArUco custom object(s) to {output_path}")


if __name__ == "__main__":
    main()
