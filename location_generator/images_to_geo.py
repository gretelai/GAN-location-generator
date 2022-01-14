import argparse
import cv2 as cv
import geopy.distance
import numpy as np
import os
import pandas as pd
import pathlib
from tqdm import tqdm

box_side_km = 1.0
image_pixels = 256


def find_points(img_path: str) -> list:
    """ Return the offsets of magenta-colored pixels in image """
    image = cv.imread(img_path)
    lower = np.array([225, 0, 225])
    upper = np.array([255, 5, 255])
    shape_mask = cv.inRange(image, lower, upper)
    #cv.imshow(shape_mask)
    pixel_x, pixel_y = np.where(shape_mask > 0)
    return pixel_x, pixel_y


def point_to_geo(center_lat: float, center_lon: float, point_x: float, point_y: float) -> list:
    """ Convert pixel offset and center points to lat/lon coordinates"""
    # Center map around current lat / lon points
    d = geopy.distance.distance(kilometers=np.sqrt(box_side_km ** 2 + box_side_km ** 2))
    ne = d.destination(point=geopy.Point(center_lat, center_lon), bearing=315)

    pixel_to_km = box_side_km * 2 / image_pixels

    lat_d = geopy.distance.distance(kilometers=pixel_to_km * point_x)
    lon_d = geopy.distance.distance(kilometers=pixel_to_km * point_y)
    point_lat = lat_d.destination(point=ne, bearing=180).latitude
    point_lon = lon_d.destination(point=ne, bearing=90).longitude
    return [point_lat, point_lon]


def main():
    # Add argument parser
    parser = argparse.ArgumentParser(description='Create location data set from synthetic images')
    parser.add_argument('--image_path', action="store", dest='image_path', type=str,
                        required=True, help=f'Provide location for synthetic images')
    parser.add_argument('--name', action="store", dest='name', type=str, default='locations.csv',
                        required=False, help=f'File to store locations to (csv)')
    args = parser.parse_args()

    geo_points = []
    print(f"Searching {args.image_path}")
    for image_path in tqdm(pathlib.Path(args.image_path).glob('*.png')):
        image_path = str(image_path)
        image_metadata = os.path.basename(image_path).split("_")
        lat = float(image_metadata[1])
        lon = float(image_metadata[2])

        x_offsets, y_offsets = find_points(image_path)

        for idx in range(len(x_offsets)):
            point_lat, point_lon = point_to_geo(lat, lon, x_offsets[idx], y_offsets[idx])
            geo_points.append({'latitude': point_lat, 'longitude': point_lon, 'source': image_path})
    df = pd.DataFrame(geo_points)
    df.to_csv(f'{args.name}', index=False)
    print(df)


if __name__ == "__main__":
    main()
