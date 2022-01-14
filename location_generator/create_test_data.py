import argparse
import folium
import io
import geopy.distance
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from shapely.geometry import Point
from tqdm import tqdm

box_side_km = 1.0


def add_noise():
    return (np.random.rand(1) / 100.0)[0]


def plot_point(lat: float, lon: float, idx: int, image_dir: Path):
    d = geopy.distance.distance(kilometers=np.sqrt(box_side_km ** 2 + box_side_km ** 2))
    ne = d.destination(point=geopy.Point(lat, lon), bearing=45)
    sw = d.destination(point=geopy.Point(lat, lon), bearing=225)

    pixel_size = 683

    m = folium.Map(location=[lat, lon], zoom_start=11, zoom_control=False, height=pixel_size, width=pixel_size)
    m.fit_bounds([[sw.latitude, sw.longitude], [ne.latitude, ne.longitude]])

    img_data = m._to_png(5)
    img = Image.open(io.BytesIO(img_data))

    # Setting the points for cropped image (remove border and controls)
    left = 85
    top = 85
    right = 512 + 85
    bottom = 512 + 85

    # Cropped image of above dimension
    # (It will not change original image)
    img = img.crop((left, top, right, bottom))
    img.save(image_dir / f'{idx:04d}_{lat:.6f}_{lon:.6f}_map.png')


def main():
    # Add argument parser
    parser = argparse.ArgumentParser(description='Example: python create_test_data.py --lat 35.652832 '
                                                 '--lon 139.839478 --name Tokyo')
    parser.add_argument('--lat', action="store", dest='lat', type=float, default=35.652832,
                        required=True, help=f'Latitude center point for grid (dd)')
    parser.add_argument('--lon', action="store", dest='lon', type=float, default=139.839478,
                        required=True, help=f'Longitude center point for grid (dd)')
    parser.add_argument('--grid_count', action="store", dest='grid_count', type=int, default=10,
                        required=False, help=f'Grid size to build (in km)')
    parser.add_argument('--name', action="store", dest='name', type=str, default='city_grid',
                        required=True, help=f'Data set name')
    args = parser.parse_args()

    image_dir = Path(f'datasets/{args.name}')
    image_dir.mkdir(parents=True, exist_ok=True)

    # Location
    location_lat = args.lat
    location_lon = args.lon

    # Define a general distance object, initialized with a distance of box_side_km
    grid_count = 15
    d = geopy.distance.distance(kilometers=np.sqrt(box_side_km ** 2 + box_side_km ** 2) * grid_count)
    ne = d.destination(point=geopy.Point(location_lat, location_lon), bearing=45)
    sw = d.destination(point=geopy.Point(location_lat, location_lon), bearing=225)
    box = [sw.longitude, ne.longitude, sw.latitude, ne.latitude]

    # Build canvas grid
    points = []
    canvas = []
    for y in np.arange(box[0], box[1], abs(box[1] - box[0]) / grid_count):
        for x in np.arange(box[2], box[3], abs(box[3] - box[2]) / grid_count):
            points.append(Point(y + add_noise(), x + add_noise()))
            canvas.append({'lon': y, 'lat': x})
    locations = pd.DataFrame(canvas)

    for idx in tqdm(range(len(locations))):
        plot_point(locations.loc[idx][1], locations.loc[idx][0], idx, image_dir)


if __name__ == "__main__":
    main()
