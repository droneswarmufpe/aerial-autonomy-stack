#!/usr/bin/env python3

import sys
import xml.etree.ElementTree as ET

def extract_spherical_coordinates(sdf_file):
    # Extract spherical coordinates from an SDF world file.
    # Returns: lat,lon,elev,0
    try:
        tree = ET.parse(sdf_file)
        root = tree.getroot()
        
        # Find the spherical_coordinates element
        spherical_coords = root.find('.//spherical_coordinates')
        
        if spherical_coords is None:
            print("0,0,0,0", file=sys.stderr)
            sys.exit(1)
        
        # Extract values
        latitude = spherical_coords.findtext('latitude_deg', '0')
        longitude = spherical_coords.findtext('longitude_deg', '0')
        elevation = spherical_coords.findtext('elevation', '0')
        
        # Format as: lat,lon,elev,0
        print(f"{latitude},{longitude},{elevation},0")
        
    except Exception as e:
        print(f"Error parsing SDF file: {e}", file=sys.stderr)
        print("0,0,0,0")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: extract_spherical_coords.py <sdf_file>", file=sys.stderr)
        sys.exit(1)
    
    extract_spherical_coordinates(sys.argv[1])
