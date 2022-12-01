#  Copyright 2021 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
#  IARAI licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License. You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import tarfile
import tempfile
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union
import argparse
#import logging
#import tqdm


def untar_files(files: List[Union[str, Path]], destination: Optional[str] = None):
    """Untar files to a destination repo."""
    #pbar = tqdm.tqdm(files, total=len(files))
    for f in files:
        print(str(f)) #pbar.set_description(str(f))
        with tarfile.open(f, "r") as tar:
            if destination is not None:
                Path(destination).mkdir(exist_ok=True)
                tar.extractall(path=destination)
            else:
                with tempfile.TemporaryDirectory() as temp_dir:
                    tar.extractall(path=temp_dir)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parser for CLI arguments to run model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    file_list = ["ANTWERP.tar.gz", "BANGKOK.tar.gz", "BARCELONA.tar.gz",
                 "BERLIN.tar.gz", "CHICAGO.tar.gz", "ISTANBUL.tar.gz",
                 "MELBOURNE.tar.gz", "MOSCOW.tar.gz", "NEWYORK.tar.gz",
                 "VIENNA.tar.gz"]
    dest_path = "./raw"

    parser.add_argument("--file_list", default=file_list, required=False,
                        help="Files to untar.")
    parser.add_argument("--dest_path", default=dest_path, required=False,
                        help="Path to untarred destionation folder.")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    untar_files(args.file_list, args.dest_path)
