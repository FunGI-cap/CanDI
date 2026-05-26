import os
import subprocess

import anndata as ad
import pandas as pd
import polars as pl
from tqdm import tqdm

from ._database import CancerDataNamespace

LATEST_VERSION = "26Q1"
FILES_URL = 'https://depmap.org/portal/api/download/files'


class DepMapAPI:
    """
    Placeholder for future API-based data retrieval methods.
    Currently, data is expected to be available as local CSV files.
    """
    def __init__(self, save_dir, version=LATEST_VERSION):
        self.version = version
        self.save_dir = save_dir + f"/{version}"
        self._files_table = pd.read_csv(FILES_URL)

    def _list_urls(self, subset_pattern=None, query_column='release'):
        urls = self._files_table.copy()
        if subset_pattern:
            urls = urls.query(
                f'{query_column}.str.contains("{subset_pattern}")'
            ).set_index('filename')
        else:
            urls = urls.set_index('filename')
        return urls
        
    def _list_depmap_urls(self):
        urls = self._list_urls(subset_pattern=f"DepMap Public {self.version}", query_column='release')
        return urls
    
    def _download_dataset(self, urls, gzip):

        urls_dict = urls.loc[:, 'url'].to_dict()

        pbar = tqdm(
            total=len(urls_dict), 
            desc="Downloading datasets",
        )
        
        for filename, url in urls_dict.items():
            save_path = os.path.join(self.save_dir, filename)
            pbar.update(1)
            #TODO add checksum verification to ensure file integrity after download
            if os.path.exists(save_path + ".gz"):
                print(f"\t{filename}.gz already exists, skipping download.")
            elif os.path.exists(save_path):
                print(f"\t{filename} already exists, skipping download.")
            else:
                print(f"\t{filename}...")
                # use wget and gzip to download and save the file
                os.makedirs(self.save_dir, exist_ok=True)
                subprocess.run(["wget", "-q", url, "-O", save_path], check=True)
                if gzip:
                    subprocess.run(["gzip", "-f", save_path], check=True)
        
        pbar.close()
    
    def download_depmap_all(self, gzip=True):
        """Download all datasets for the specified version."""
        self._download_dataset(self._list_depmap_urls(),gzip=gzip)
        
    def download_depmap_essential(self, gzip=True):
        """Download only essential datasets for the specified version."""
        urls = self._list_depmap_urls()
        essential_files = [
            "Model.csv",
            "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv", # renamed in 25Q3!
            "OmicsSomaticMutations.csv",
            "OmicsSomaticMutationsMatrixDamaging.csv",
            "OmicsCNGeneWGS.csv",
            "CRISPRGeneDependency.csv",
            "CRISPRGeneEffect.csv",
            "OmicsCNSegmentsWGS.csv"
        ]
        essential_urls = urls.loc[essential_files,:]
        
        self._download_dataset(essential_urls, gzip=gzip)

    def download_subset(self, subset_pattern, query_column='release', gzip=True):
        """Download a subset of datasets matching the given pattern.
        This is useful for downloading specific datasets, e.g. 
        `Harmonized PRISM` query for the PRISM drug response dataset.
        """
        urls = self._list_urls(subset_pattern=subset_pattern, query_column=query_column)
        if urls.empty:
            print(f"No datasets found matching pattern: {subset_pattern}")
            return
        self._download_dataset(urls, gzip=gzip)



class DepMapData:
    """
    Data handler for DepMap datasets.
    
    Supports lazy loading of datasets into memory and unloading them when no longer needed.
    Provides attribute-style access to datasets (e.g., obj.data.Model).
    """

    class DataNamespace(CancerDataNamespace):
        """Namespace object for dataset access under `.data`."""

        # DepMap main datasets
        Model: pd.DataFrame
        OmicsExpression: pd.DataFrame
        OmicsSomaticMutations: pd.DataFrame
        OmicsSomaticMutationsMatrixDamaging: pd.DataFrame
        OmicsCNGeneWGS: pd.DataFrame
        CRISPRGeneDependency: pd.DataFrame
        CRISPRGeneEffect: pd.DataFrame

    def __init__(self, data_dir, version=LATEST_VERSION):
        self.data_dir = data_dir
        self.version = version
        self._datasets = {}  # Holds loaded datasets in memory
        self._paths = self._get_dataset_paths()
        self._check_paths_exist()
        self.data = self.DataNamespace(self)

    def __repr__(self):
        """Display object info when called interactively."""
        info = [f"DepMapData(version={self.version})"]
        info.append("")
        info.append("Available datasets:")
        for dataset in self.list_available():
            status = "Loaded" if dataset in self._datasets else "Not loaded"
            info.append(f" - {dataset}: {status}")
        return "\n".join(info)

    def _get_dataset_paths(self):
        """Define paths for available datasets."""
        base = os.path.join(self.data_dir, self.version)
        return {
            "Model": os.path.join(base, "Model.csv.gz"),
            "OmicsExpression": os.path.join(base, "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv.gz"), # renamed in 25Q3!
            "OmicsSomaticMutations": os.path.join(base, "OmicsSomaticMutations.csv.gz"),
            "OmicsSomaticMutationsMatrixDamaging": os.path.join(base, "OmicsSomaticMutationsMatrixDamaging.csv.gz"),
            "OmicsCNGeneWGS": os.path.join(base, "OmicsCNGeneWGS.csv.gz"),
            "OmicsCNSegmentsWGS": os.path.join(base, "OmicsCNSegmentsWGS.csv.gz"),
            "CRISPRGeneDependency": os.path.join(base, "CRISPRGeneDependency.csv.gz"),
            "CRISPRGeneEffect": os.path.join(base, "CRISPRGeneEffect.csv.gz"),
        }

    def _check_paths_exist(self):
        """Check that all expected dataset files exist, otherwise raise an error."""
        missing = [name for name, path in self._paths.items() if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                f"The following dataset files are missing for version {self.version}: {', '.join(missing)}"
            )

    def load(self, name, inplace=True, engine='polars', **kwargs):
        """
        Load a dataset into memory.

        Parameters
        ----------
        name : str
            Name of the dataset to load.
        inplace : bool, default True
            - If True: stores the dataset inside the object (retrievable via .data.<name> or .get()).
            - If False: returns the dataset as a DataFrame without storing.
        kwargs : dict
            Additional arguments passed to the selected CSV reader.
        """
        if name not in self._paths:
            raise ValueError(f"Dataset {name} is not defined for version {self.version}.")
        if engine not in {'pandas', 'polars'}:
            raise ValueError("engine must be either 'pandas' or 'polars'.")

        if inplace and name in self._datasets:
            return self._datasets[name]  # Already loaded

        path = self._paths[name]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        def _read_csv(file_path, **read_kwargs):
            if engine == 'pandas':
                return pd.read_csv(file_path, **read_kwargs)

            index_col = read_kwargs.pop("index_col", None)
            data = pl.read_csv(file_path, **read_kwargs).to_pandas()
            if index_col is not None:
                if isinstance(index_col, int):
                    data = data.set_index(data.columns[index_col])
                else:
                    data = data.set_index(index_col)
            return data

        # Default loading logic
        if name == "Model":
            df = _read_csv(path, **kwargs).set_index("ModelID")
            data = df.copy()

        elif name in {
            "CRISPRGeneDependency", "CRISPRGeneEffect"
            }:
            df = _read_csv(path, index_col=0, **kwargs)
            df.columns = df.columns.str.split(" ").str[0]

            data = df.copy()
        
        elif name in {
            "OmicsExpression","OmicsCNGeneWGS",
            "OmicsSomaticMutationsMatrixDamaging",
            }:
            df = _read_csv(path, index_col=0, **kwargs).set_index("ModelID")
            # only keep columns with " "
            df = df.loc[:, df.columns.str.contains(" ")].copy()
            df.columns = df.columns.str.split(" ").str[0]

            data = df.copy()

        elif name in {
            "OmicsSomaticMutations",
            }:
            data = _read_csv(path, index_col=0, **kwargs).set_index("ModelID")

        else:
            data = _read_csv(path, **kwargs)
        
        if inplace:
            self._datasets[name] = data
            return data
        else:
            return data

    def unload(self, name):
        """Remove dataset from memory."""
        if name in self._datasets:
            del self._datasets[name]

    def load_all(self):
        """Load all available datasets into memory."""
        for name in self._paths.keys():
            self.load(name, inplace=True, low_memory=False)

    def list_available(self):
        """List all available datasets for this version."""
        return list(dict.fromkeys([*self._paths.keys(), *self._datasets.keys()]))

    def add_dataset(self, name, dataset, overwrite=False):
        """Add a user-provided dataset to the in-memory namespace."""
        self.data.add(name=name, dataset=dataset, overwrite=overwrite)

    def get(self, name):
        """Retrieve dataset if already loaded, otherwise prompt to load it."""
        if name not in self._datasets:
            response = input(f"Dataset {name} is not loaded. Would you like to load it? (Y/N): ")
            if response.strip().lower() == 'y':
                self.load(name)
            else:
                raise RuntimeError(f"Dataset {name} is not loaded. Call `.load('{name}')` first.")
        return self._datasets[name]
