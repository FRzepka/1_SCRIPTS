import pandas as pd
import os
import pathlib
from tqdm import tqdm



######## Instructions ###########
"""
Enter the original path and the target path. 
Folder structure is copied and csv is transformed into parquet.

src_path = Source / original data
dest_path = Destination / target data

"""


class ParquetTransformer:
    def __init__(self, src_dir_path: str, dest_dir_path: str, separator: str):
        self.src_dir_path = pathlib.Path(src_dir_path).absolute()
        self.dest_dir_path = pathlib.Path(dest_dir_path).absolute()
        self.separator = separator
        assert self.src_dir_path.exists(), f"Directory {self.src_dir_path} does not exist."


    def generate_parquet_copy(self) -> None:
        print(f"Given directory: {self.src_dir_path}")
        self._generate_mirror_dir_tree()
        assert self.dest_dir_path.exists(), f"Directory {self.dest_dir_path} could not be created."
        print(f"Generated mirror directory tree in {self.dest_dir_path}")
        self._csv_to_parquet_mirror()
        print(f"Generated parquet files in {self.dest_dir_path}")
        
    def _generate_mirror_dir_tree(self) -> None:
        dir_list = [self.src_dir_path]
        for root, dirs, _ in os.walk(self.src_dir_path):
            for dir in dirs:
                dir_list.append(os.path.join(root, dir))
        dir_list = [pathlib.Path(dir) for dir in dir_list]
        dir_list = [pathlib.Path(str(dir).replace(str(self.src_dir_path), str(self.dest_dir_path), 1)) for dir in dir_list]
        for dir in dir_list:
            os.makedirs(dir, exist_ok=True)

    def _csv_to_parquet_mirror(self) -> None:
        csv_files = [os.path.join(root, file) for root, _, files in os.walk(self.src_dir_path) for file in files if file.endswith(".csv")]
        for csv_path in tqdm(csv_files, desc="Converting CSV to Parquet"):
            csv_path = pathlib.Path(csv_path)
            parquet_path = pathlib.Path(str(csv_path).replace(str(self.src_dir_path), str(self.dest_dir_path), 1)).with_suffix('.parquet')
            if not parquet_path.exists():
                df = pd.read_csv(csv_path, sep=self.separator)
                df.to_parquet(parquet_path)
            else:
                print(f"Parquet file {parquet_path} already exists. Skipping...")

if __name__ == "__main__":
    src_path = r'c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\4_Work_packages\3_State_estimation\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650'
    dest_path = r'c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\4_Work_packages\3_State_estimation\01_LFP\00_Data\Versuch_18650_standart\MGFarm_18650_parquet'
    
    
    separator = ';'  # Trennzeichen für die CSV-Dateien
    transformer = ParquetTransformer(src_path, dest_path, separator)
    transformer.generate_parquet_copy()
