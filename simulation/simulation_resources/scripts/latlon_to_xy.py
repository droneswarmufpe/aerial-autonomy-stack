#!/usr/bin/env python3
"""
Convert geographic (lat/lon) coordinates to local Cartesian (x/y) coordinates
relative to a reference/origin point, using an equirectangular approximation
(accurate for short distances, e.g. within a single simulation world).

CLI usage:
    python3 latlon_to_xy.py <lat> <lon> <lat_ref> <lon_ref>

Prints "X Y" (space separated, in meters) to stdout, so it can be
consumed easily from shell scripts, e.g.:

    read -r X Y <<< "$(python3 latlon_to_xy.py 48.81 7.85 48.80 7.85)"

Can also be imported as a module:

    from latlon_to_xy import latlon_to_xy
    x, y = latlon_to_xy(lat, lon, lat_ref, lon_ref)
"""

import math
import sys

EARTH_RADIUS_M = 6371000.0


def latlon_to_xy(lat, lon, lat_ref, lon_ref, earth_radius=EARTH_RADIUS_M):
    """
    Convert (lat, lon) to local (x, y) meters relative to (lat_ref, lon_ref)
    using an equirectangular projection.

    X corresponds to the East-West offset (positive East).
    Y corresponds to the North-South offset (positive North).
    """
    d_lat = math.radians(lat - lat_ref)
    d_lon = math.radians(lon - lon_ref)

    x = earth_radius * d_lon * math.cos(math.radians(lat_ref))
    y = earth_radius * d_lat

    return x, y


def main():
    if len(sys.argv) != 5:
        print(
            f"Usage: {sys.argv[0]} <lat> <lon> <lat_ref> <lon_ref>",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        lat, lon, lat_ref, lon_ref = (float(v) for v in sys.argv[1:5])
    except ValueError:
        print("ERROR: lat/lon arguments must be numeric.", file=sys.stderr)
        sys.exit(1)

    x, y = latlon_to_xy(lat, lon, lat_ref, lon_ref)
    print(f"{x} {y}")


if __name__ == "__main__":
    main()