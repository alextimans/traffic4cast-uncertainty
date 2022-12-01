import argparse
import random
import os
import numpy as np
from pathlib import Path
from typing import Union
from data.data_layout import CITY_NAMES, CITY_TRAIN_ONLY


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parser for CLI arguments to run model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cities", nargs="+", type=str, default=[None, None, None, None], required=False,
                        help="Limit cities for which to generate test index files.")
    parser.add_argument("--samp_size", type=int, default=100, required=False,
                        help="#samples for test run.")
    parser.add_argument("--antwerp_2019", type=str, default="False", required=False, choices=["True", "False"],
                        help="'Boolean' if sample for 2019+2020 data only for Antwerp. Otherwise sampled only from 2020 data.")
    parser.add_argument("--test_samp_path", type=str, default=None, required=False,
                        help="Test sample indices file path.")
    parser.add_argument("--test_samp_name", type=str, default="test_indices.txt", required=False,
                        help="Test sample indices file name.")
    return parser


def get_random_sample_idx(citylist: list, samp_size: int, antwerp_2019: str,
                          test_samp_path: str, test_samp_name: str):

    cities = [city for city in CITY_NAMES if city not in CITY_TRAIN_ONLY]
    if not any(citylist):
        citylist = cities
    cities = list(set(cities).intersection(set(citylist)))

    for city in cities:
        if city == "ANTWERP":
            if eval(antwerp_2019):
                sub_idx = random.sample(range(0, 51840), samp_size) # antwerp 2019 + 2020
            else:
                sub_idx = random.sample(range(25920, 51840), samp_size) # antwerp 2020
        else:
            sub_idx = random.sample(range(0, 25920), samp_size) # other cities 2020


        save_file_to_folder(file=sub_idx, filename=test_samp_name,
                            folder_dir=os.path.join(test_samp_path, city))


def save_file_to_folder(file = None, filename: str = None,
                        folder_dir: Union[Path, str] = None, **kwargs):

    folder_path = Path(folder_dir) if isinstance(folder_dir, str) else folder_dir
    folder_path.mkdir(exist_ok=True, parents=True)
    np.savetxt(os.path.join(folder_path, f"{filename}.txt"), file, **kwargs)


def main():
    parser = create_parser()
    args = parser.parse_args()
    get_random_sample_idx(citylist=args.cities,
                          samp_size=args.samp_size,
                          antwerp_2019=args.antwerp_2019,
                          test_samp_path=args.test_samp_path,
                          test_samp_name=args.test_samp_name)


if __name__ == "__main__":
    main()
