import json
import os
from typing import Dict, Generator, List

from x_ray.inference.mercantile import geojson_bounds, tiles


def str_to_dict(dict_strs: List[str]) -> Generator:
    for dict_str in dict_strs:
        d: dict = json.loads(dict_str)
        yield d


def get_tiles(geojson: dict, zooms: List[str] = [15], truncate: bool = True) -> Generator:  
    geo_bounds = geojson_bounds(geojson)
    west = geo_bounds.west
    south = geo_bounds.south
    east = geo_bounds.east
    north = geo_bounds.north

    tiles_gen = tiles(west, south, east, north, zooms, truncate)
    for tile in tiles_gen:
        yield tile


if __name__ == "__main__":
    geojson_path: str = os.environ["GEOJSON_PATH"]
    with open(geojson_path) as f:
        geometry: Dict = json.load(f)
    geometry_str: str = json.dumps(geometry)
    geojson_dicts: Generator = str_to_dict([geometry_str])
    for geojson_dict in list(geojson_dicts):
        tiles_gen: Generator = get_tiles(geojson=geometry)
        tiles_sub_list = list(tiles_gen)
        print(f"Number of tiles for geojson {geojson_path}: {len(tiles_sub_list)}.")

