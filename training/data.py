import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
)
DATA_DIR = Path("data")
RAW_ZIP = DATA_DIR / "raw.zip"
CSV_PATH = DATA_DIR / "credit_default.csv"
TARGET = "default"
DATASET_NAME = "uci-default-of-credit-card-clients"


def _load_raw() -> pd.DataFrame:
    if not RAW_ZIP.exists():
        DATA_DIR.mkdir(exist_ok=True)
        with urlopen(SOURCE_URL, timeout=60) as response:
            RAW_ZIP.write_bytes(response.read())
    with zipfile.ZipFile(RAW_ZIP) as archive:
        content = archive.read(archive.namelist()[0])
    return pd.read_excel(io.BytesIO(content), header=1)


def load_dataset() -> pd.DataFrame:
    """Единая точка загрузки. Пол убран: в кредитном решении его использовать нельзя."""
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)
    frame = _load_raw()
    frame.columns = [c.lower() for c in frame.columns]
    frame = frame.rename(columns={"default payment next month": TARGET})
    frame = frame.drop(columns=["id", "sex"])
    frame.to_csv(CSV_PATH, index=False)
    return frame
