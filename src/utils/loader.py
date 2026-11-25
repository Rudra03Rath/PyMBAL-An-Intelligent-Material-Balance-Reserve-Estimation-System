"""
PyMBAL - Data Loader
"""

import pandas as pd
from pathlib import Path


class DataLoader:

    def __init__(self, filepath):

        self.filepath = Path(filepath)

    def load(self):

        if not self.filepath.exists():

            raise FileNotFoundError(
                f"File not found:\n{self.filepath}"
            )

        extension = self.filepath.suffix.lower()

        if extension == ".csv":

            df = pd.read_csv(
                self.filepath
            )

        elif extension in [".xlsx", ".xls"]:

            df = pd.read_excel(
                self.filepath
            )

        else:

            raise ValueError(
                "Unsupported file format.\n"
                "Supported formats: CSV, XLSX, XLS"
            )

        if df.empty:

            raise ValueError(
                "The input file contains no data."
            )

        print(
            f"\n✓ Loaded {len(df)} records."
        )

        print(
            f"✓ File: {self.filepath.name}"
        )

        return df