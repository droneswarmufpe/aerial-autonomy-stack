#!/usr/bin/env python3
"""Generate a custom_objects_config.json with vehicles randomly scattered inside an area.

The world's origin (lat/lon/elevation) is read from the <spherical_coordinates>
tag of a world .sdf file. An area (rectangle, circle, or an arbitrary polygon) is
then defined around an input lat/lon coordinate (or, for polygon, a list of
vertex lat/lon coordinates), vehicles are placed at random positions (and random
yaw) inside that area, and the result is written as a custom_objects_config.json
using each object's "global_position" (lat/lon/alt) and "attitude" (roll/pitch/yaw),
matching the schema read by _create_custom_objects_world.sh.

Examples:
  # 12 vehicles randomly placed in a 150m radius circle around a coordinate
  ./generate_custom_objects_config.py \\
      --world ../simulation/simulation_resources/simulation_worlds/imav.sdf \\
      circle --lat 48.803 --lon 7.854 --radius 150 \\
      --num-objects 12 -o ../simulation/simulation_resources/custom_objects/custom_objects_config_generated.json

  # Vehicles inside a 200x80m rectangle, rotated 30 degrees, centered on a coordinate
  ./generate_custom_objects_config.py \\
      --world ../simulation/simulation_resources/simulation_worlds/imav.sdf \\
      rectangle --lat 48.804 --lon 7.855 --width 200 --length 80 --heading 30

  # Vehicles inside an arbitrary polygon, given by its vertices (walked either
  # clockwise or counter-clockwise, doesn't matter which)
  ./generate_custom_objects_config.py --world ../simulation/simulation_resources/simulation_worlds/imav.sdf \\
  polygon --vertices 48.80570456695289,7.84961762660713 \\      
    48.810842794362294,7.847237082007214 \\
    48.81380786901547,7.861214602971546 \\
    48.80863939598765,7.864039635567677
"""
import argparse
import json
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# Mean Earth radius, matching simulation_resources/custom_objects/latlon_to_xy.py
# so areas sampled here line up with how the world loader converts lat/lon back
# to local x/y. Good enough for the flat-earth approximation used here, valid
# for areas up to a few kilometers across.
EARTH_RADIUS_M = 6371000.0

DEFAULT_MODELS = [
    # "toyota_hilux",
    "vltt_a",
    "panhard_vbl",
    "renault_ccfm",
    "renault_gbc180",
    "arquus_vt4",
    "jeep",
    # "subaru",
]


def parse_world_info(world_path: Path):
    """Extract the world name and its origin latitude/longitude/elevation.

    The name comes from <world name="..."> and the origin from the
    <spherical_coordinates> tag.
    """
    tree = ET.parse(world_path)
    root = tree.getroot()

    world_elem = root.find(".//world")
    world_name = world_elem.get("name") if world_elem is not None and world_elem.get("name") else world_path.stem

    sph = root.find(".//spherical_coordinates")
    if sph is None:
        raise ValueError(f"No <spherical_coordinates> tag found in {world_path}")

    lat = sph.findtext("latitude_deg")
    lon = sph.findtext("longitude_deg")
    elev = sph.findtext("elevation")
    if lat is None or lon is None or elev is None:
        raise ValueError(
            f"<spherical_coordinates> in {world_path} is missing "
            "latitude_deg, longitude_deg or elevation"
        )

    frame = (sph.findtext("world_frame_orientation") or "ENU").strip().upper()
    if frame != "ENU":
        print(
            f"Warning: world_frame_orientation is '{frame}', expected 'ENU'. "
            "Local x/y offsets computed here assume X=East, Y=North."
        )

    return world_name, float(lat), float(lon), float(elev)


def latlon_to_local_xy(lat, lon, origin_lat, origin_lon):
    """Equirectangular projection relative to the world origin (X=East meters, Y=North meters)."""
    origin_lat_rad = math.radians(origin_lat)
    y = math.radians(lat - origin_lat) * EARTH_RADIUS_M
    x = math.radians(lon - origin_lon) * EARTH_RADIUS_M * math.cos(origin_lat_rad)
    return x, y


def local_xy_to_latlon(x, y, origin_lat, origin_lon):
    """Inverse of latlon_to_local_xy: local meters (X=East, Y=North) back to lat/lon."""
    origin_lat_rad = math.radians(origin_lat)
    lat = origin_lat + math.degrees(y / EARTH_RADIUS_M)
    lon = origin_lon + math.degrees(x / (EARTH_RADIUS_M * math.cos(origin_lat_rad)))
    return lat, lon


def sample_in_rectangle(width, length, heading_deg, rng):
    """Uniform random point in a width x length rectangle, then rotated by heading_deg."""
    u = rng.uniform(-width / 2, width / 2)
    v = rng.uniform(-length / 2, length / 2)
    theta = math.radians(heading_deg)
    x = u * math.cos(theta) - v * math.sin(theta)
    y = u * math.sin(theta) + v * math.cos(theta)
    return x, y


def sample_in_circle(radius, min_radius, rng):
    """Uniform random point in an annulus between min_radius and radius."""
    inner_frac = (min_radius / radius) ** 2 if radius > 0 else 0.0
    r = radius * math.sqrt(rng.uniform(inner_frac, 1.0))
    theta = rng.uniform(0, 2 * math.pi)
    return r * math.cos(theta), r * math.sin(theta)


def point_in_polygon(x, y, poly_xy):
    """Ray-casting point-in-polygon test. Works for either winding order."""
    inside = False
    n = len(poly_xy)
    xj, yj = poly_xy[-1]
    for xi, yi in poly_xy:
        if (yi > y) != (yj > y):
            x_intersect = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < x_intersect:
                inside = not inside
        xj, yj = xi, yi
    return inside


def sample_in_polygon(poly_xy, rng, max_attempts=10000):
    """Uniform random point inside poly_xy via rejection sampling on its bounding box."""
    min_x = min(p[0] for p in poly_xy)
    max_x = max(p[0] for p in poly_xy)
    min_y = min(p[1] for p in poly_xy)
    max_y = max(p[1] for p in poly_xy)
    for _ in range(max_attempts):
        x = rng.uniform(min_x, max_x)
        y = rng.uniform(min_y, max_y)
        if point_in_polygon(x, y, poly_xy):
            return x, y
    raise RuntimeError(
        "Failed to sample a point inside the polygon after "
        f"{max_attempts} attempts. Check that --vertices describes a valid, "
        "non-degenerate simple polygon (it may be extremely thin/sliver-shaped)."
    )


def latlon_pair(value):
    """argparse type for a 'lat,lon' vertex string."""
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid lat,lon pair: '{value}'. Expected format: LAT,LON")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid lat,lon pair: '{value}'. Expected format: LAT,LON")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--world", required=True, type=Path,
        help="Path to the world .sdf file to read the origin lat/lon/elevation from "
             "(e.g. simulation/simulation_resources/simulation_worlds/imav.sdf)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output custom_objects_config.json path. Defaults to "
             "simulation/simulation_resources/custom_objects/custom_objects_config_generated.json",
    )
    parser.add_argument("--num-objects", "-n", type=int, default=10, help="Number of vehicles to spawn")
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help=f"Model names to randomly choose from (default: {DEFAULT_MODELS})",
    )
    parser.add_argument("--alt", type=float, default=0.0, help="Base global_position altitude in meters")
    parser.add_argument(
        "--alt-jitter", type=float, default=0.0,
        help="Random +/- jitter in meters applied to --alt for each object",
    )
    parser.add_argument("--roll", type=float, default=0.0, help="Fixed attitude.roll in degrees for every object")
    parser.add_argument(
        "--pitch", type=float, default=0.0,
        help="Fixed attitude.pitch in degrees for every object (0 matches the "
             "upright orientation used by the existing vehicle models)",
    )
    parser.add_argument(
        "--yaw-min", type=float, default=0.0, help="Minimum random attitude.yaw (degrees) for each object"
    )
    parser.add_argument(
        "--yaw-max", type=float, default=360.0, help="Maximum random attitude.yaw (degrees) for each object"
    )
    parser.add_argument(
        "--static", choices=["true", "false", "random"], default="false",
        help="Whether spawned objects are static. 'random' rolls per-object using --static-ratio",
    )
    parser.add_argument(
        "--static-ratio", type=float, default=0.5,
        help="Fraction of objects marked static when --static=random",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible layouts")

    subparsers = parser.add_subparsers(dest="shape", required=True, help="Shape of the spawn area")

    rectangle = subparsers.add_parser("rectangle", help="Rectangular spawn area")
    rectangle.add_argument("--lat", type=float, required=True, help="Center latitude (deg)")
    rectangle.add_argument("--lon", type=float, required=True, help="Center longitude (deg)")
    rectangle.add_argument("--width", type=float, required=True, help="East-West extent in meters (before --heading)")
    rectangle.add_argument("--length", type=float, required=True, help="North-South extent in meters (before --heading)")
    rectangle.add_argument("--heading", type=float, default=0.0, help="Rectangle rotation in degrees, clockwise from North")

    circle = subparsers.add_parser("circle", help="Circular spawn area")
    circle.add_argument("--lat", type=float, required=True, help="Center latitude (deg)")
    circle.add_argument("--lon", type=float, required=True, help="Center longitude (deg)")
    circle.add_argument("--radius", type=float, required=True, help="Radius in meters")
    circle.add_argument("--min-radius", type=float, default=0.0, help="Inner exclusion radius in meters (ring spawn)")

    polygon = subparsers.add_parser(
        "polygon", help="Arbitrary polygon spawn area defined by its vertices"
    )
    polygon.add_argument(
        "--vertices", nargs="+", type=latlon_pair, required=True, metavar="LAT,LON",
        help="Polygon vertices as 'lat,lon' pairs, walked in order around the "
             "perimeter (clockwise or counter-clockwise, either works), at least 3 required",
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    rng = random.Random(args.seed)

    world_name, origin_lat, origin_lon, origin_elev = parse_world_info(args.world)

    poly_xy = None
    if args.shape == "polygon":
        if len(args.vertices) < 3:
            parser.error("polygon requires at least 3 --vertices")
        poly_xy = [latlon_to_local_xy(lat, lon, origin_lat, origin_lon) for lat, lon in args.vertices]
        center_x = sum(p[0] for p in poly_xy) / len(poly_xy)
        center_y = sum(p[1] for p in poly_xy) / len(poly_xy)
    else:
        center_x, center_y = latlon_to_local_xy(args.lat, args.lon, origin_lat, origin_lon)

    objects = []
    for i in range(1, args.num_objects + 1):
        if args.shape == "rectangle":
            dx, dy = sample_in_rectangle(args.width, args.length, args.heading, rng)
            x, y = center_x + dx, center_y + dy
        elif args.shape == "circle":
            dx, dy = sample_in_circle(args.radius, args.min_radius, rng)
            x, y = center_x + dx, center_y + dy
        else:
            x, y = sample_in_polygon(poly_xy, rng)

        if args.static == "random":
            static = rng.random() < args.static_ratio
        else:
            static = args.static == "true"

        lat, lon = local_xy_to_latlon(x, y, origin_lat, origin_lon)

        objects.append({
            "id": i,
            "global_position": {
                "lat": round(lat, 8),
                "lon": round(lon, 8),
                "alt": round(args.alt + rng.uniform(-args.alt_jitter, args.alt_jitter), 4),
            },
            "attitude": {
                "roll": args.roll,
                "pitch": args.pitch,
                "yaw": round(rng.uniform(args.yaw_min, args.yaw_max), 2),
            },
            "model": rng.choice(args.models),
            "static": static,
        })

    output_path = args.output
    if output_path is None:
        output_path = (
            Path(__file__).resolve().parent
            / ".." / "simulation" / "simulation_resources" / "custom_objects"
            / "custom_objects_config_generated.json"
        ).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "world": world_name,
        "origin": {"lat": origin_lat, "lon": origin_lon},
        "objects": objects,
    }
    output_path.write_text(json.dumps(output_data, indent=2) + "\n")

    center_label = "Polygon centroid" if args.shape == "polygon" else "Area center"
    print(f"World origin ({args.world}): lat={origin_lat}, lon={origin_lon}, elevation={origin_elev}m")
    print(f"{center_label} local xy: x={center_x:.2f}m (East), y={center_y:.2f}m (North)")
    print(f"Generated {len(objects)} object(s) -> {output_path}")
    print(
        "Note: point CUSTOM_OBJECTS_CONFIG_FILE in "
        "simulation/simulation_resources/simulation_worlds/_create_custom_objects_world.sh "
        "at this file (or overwrite custom_objects_config.json) to use it."
    )


if __name__ == "__main__":
    main()
