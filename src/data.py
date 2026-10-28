import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def load_data():
    dataset = load_breast_cancer()
    return dataset.data, (dataset.target == 0).astype(int)

def make_preprocessor():
    return Pipeline([('standardize', StandardScaler()), ('pca', PCA(n_components=4)), ('angles', MinMaxScaler(feature_range=(0, np.pi), clip=True))])

def load_subset_indices(path):
    """Read an original subset from disk or the committed limited-data ZIP."""
    from pathlib import Path
    from io import BytesIO
    import zipfile
    path = Path(path)
    if path.is_file():
        with np.load(path) as data:
            return data["train_idx"].copy()
    archive = Path("testing/results/qbreast_limited_data.zip")
    if archive.is_file():
        with zipfile.ZipFile(archive) as z:
            matches = [name for name in z.namelist()
                       if name.replace("\\", "/").endswith("/" + path.name)
                       or name == path.name]
            if len(matches) == 1:
                with np.load(BytesIO(z.read(matches[0]))) as data:
                    return data["train_idx"].copy()
    raise FileNotFoundError(f"Subset {path} not found. Check ZIP members with: python -m zipfile -l {archive}")
