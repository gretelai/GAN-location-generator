import argparse
import selenium
import folium
import io
import geopy.distance
import multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm


data_set = 'https://gretel-public-website.s3-us-west-2.amazonaws.com/datasets/uber_scooter_rides_usa.csv'
df = pd.read_csv(data_set)
df = df.sample(frac=1.0, random_state=42)
box_side_km = 1.0

train_a_dir = Path('datasets/trainA')
train_b_dir = Path('datasets/trainB')


def add_circle_marker(point: pd.Series, fmap: folium.Map, center_lat: float, center_lon: float):
    coord_a = geopy.Point(point.lat, point.lon)
    coord_b = geopy.Point(center_lat, center_lon)
    if geopy.distance.distance(coord_a, coord_b).km < box_side_km * 2:
        folium.CircleMarker(location=[point.lat, point.lon],
                            color='magenta',
                            radius=1,
                            weight=5).add_to(fmap)


def plot_point(lat: float, lon: float, idx: int):

    # Center map around current lat / lon points
    d = geopy.distance.distance(kilometers=np.sqrt(box_side_km ** 2 + box_side_km ** 2))
    ne = d.destination(point=geopy.Point(lat, lon), bearing=45)
    sw = d.destination(point=geopy.Point(lat, lon), bearing=225)

    pixel_size = 683
    m = folium.Map(location=[lat, lon], zoom_start=11, zoom_control=False, height=pixel_size, width=pixel_size)
    m.fit_bounds([[sw.latitude, sw.longitude], [ne.latitude, ne.longitude]])

    # Use selenium to capture screenshot of the map
    img_data = m._to_png(5)
    img = Image.open(io.BytesIO(img_data))

    # Set crop points for image to remove nav bars
    left = 85
    top = 85
    right = 512+85
    bottom = 512+85
    img = img.crop((left, top, right, bottom))

    # Encode lat/lon data in image name and save
    img.save(train_b_dir / f'{idx:04d}_{lat:.6f}_{lon:.6f}_map.png')

    # Now plot map including scooter location data
    m = folium.Map(location=[lat, lon], zoom_start=11, zoom_control=False, height=683, width=683)
    m.fit_bounds([[sw.latitude, sw.longitude], [ne.latitude, ne.longitude]])
    df.apply(add_circle_marker, fmap=m, center_lat=lat, center_lon=lon, axis=1)
    img_data = m._to_png(5)
    img = Image.open(io.BytesIO(img_data))
    img = img.crop((left, top, right, bottom))
    img.save(train_a_dir / f'{idx:04d}_{lat:.6f}_{lon:.6f}_loc.png')


def main():
    # Add argument parser
    parser = argparse.ArgumentParser(description='Create training sets for CUT model')
    parser.add_argument('--start-at', action="store", dest='start_at', type=int, default=0,
                        required=False, help=f'Choose location index to start with. Default: 0')
    parser.add_argument('--end-at', action="store", dest='end_at', type=int, default=len(df),
                        required=False, help=f'Choose location index to end at.')
    args = parser.parse_args()

    train_a_dir.mkdir(parents=True, exist_ok=True)
    train_b_dir.mkdir(parents=True, exist_ok=True)

    # Multiprocessing (does not speed up unfortunately)
    #print(f"Found {mp.cpu_count()} CPUs")
    #pool = mp.Pool(mp.cpu_count())
    #[pool.apply(plot_point, args=(df.loc[idx].lat, df.loc[idx].lon, idx)) for idx in tqdm(range(args.start_at, args.end_at, 1))]

    # Plot images to training directories
    for idx in tqdm(range(args.start_at, args.end_at, 1)):
        plot_point(df.loc[idx].lat, df.loc[idx].lon, idx)


if __name__ == "__main__":
    main()
