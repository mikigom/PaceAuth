"""Identity / distractor sources for E2 under data/datasets/<name>/ (provenance + licence in each DATASET.json).

Every loader returns DataFrame[subject, path]: `path` relative to data/datasets, `subject` unique across
datasets ("<dataset>:<id>"). Zip members are addressed as "<zip path>::<member>" and read without extraction.
`hints(name)` gives annotated 5-point landmarks (InsightFace order) per path where the dataset has them; they are only
used to pick WHICH detected face is the labelled one, never as keypoints.
"""
import zipfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DATA = Path("data/datasets")
ALIGNED = ("digiface1m/",)  # already ArcFace-style 112x112 crops -> recognizer.embed_aligned, no detector


def celeba(root=DATA, images="img_celeba"):
    """CelebA: 10,177 identities / 202,599 images. images = img_celeba (in-the-wild) | img_align_celeba."""
    df = pd.read_csv(Path(root) / "celeba/Anno/identity_CelebA.txt", sep=" ", header=None, names=["file", "id"])
    return pd.DataFrame({"subject": "celeba:" + df.id.astype(str), "path": f"celeba/{images}/" + df.file})


def ffhq(root=DATA):
    """FFHQ (256px PNG from the 1024px aligned set): 70,000 Flickr faces; identity unlabelled -> one subject per
    image (assumed distinct; FFHQ does not guarantee it)."""
    stems = sorted(p.stem for p in (Path(root) / "ffhq/images256").glob("*.png"))
    return pd.DataFrame({"subject": [f"ffhq:{s}" for s in stems], "path": [f"ffhq/images256/{s}.png" for s in stems]})


def digiface1m(root=DATA):
    """DigiFace-1M (SYNTHETIC): 109,999 identities (10K x 72 + 99,999 x 5 images), aligned 112x112 RGBA PNG,
    read from the original zips (raw/subjects_*.zip, members '<id>/<k>.png')."""
    rows = []
    for z in sorted((Path(root) / "digiface1m/raw").glob("subjects_*.zip")):
        rel = z.relative_to(root).as_posix()
        with zipfile.ZipFile(z) as f:
            rows += [(m.split("/")[0], f"{rel}::{m}") for m in f.namelist() if m.endswith(".png")]
    df = pd.DataFrame(rows, columns=["id", "path"])
    return pd.DataFrame({"subject": "digiface1m:" + df.id, "path": df.path})


def vggface2_test(root=DATA):
    """VGGFace2 test split: 500 identities / 169,396 loosely cropped images."""
    files = (Path(root) / "vggface2_test/raw/meta/test_list.txt").read_text().split()
    return pd.DataFrame({"subject": ["vggface2:" + f.split("/")[0] for f in files],
                         "path": ["vggface2_test/test/" + f for f in files]})


LOADERS = {"celeba": celeba, "ffhq": ffhq, "digiface1m": digiface1m, "vggface2_test": vggface2_test}


def load(name: str, root=DATA) -> pd.DataFrame:
    return LOADERS[name](root).sort_values("path", kind="stable").reset_index(drop=True)


@lru_cache(maxsize=16)
def _zip(path: str) -> zipfile.ZipFile:  # one handle per zip per process (worker processes each open their own)
    return zipfile.ZipFile(path)


def read_rgb(path: str, root=DATA):
    """HxWx3 uint8 RGB (alpha dropped) or None if unreadable. `path` relative to root, zip members as 'a.zip::m'."""
    if "::" in path:
        z, m = path.split("::", 1)
        try:
            buf = np.frombuffer(_zip(str(Path(root) / z)).read(m), np.uint8)
        except KeyError:
            return None
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(Path(root) / path), cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def hints(name: str, root=DATA) -> dict:
    """path -> (5,2) float32 annotated landmarks (image-left eye, image-right eye, nose, left/right mouth)."""
    root = Path(root)
    if name == "celeba":
        a = pd.read_csv(root / "celeba/Anno/list_landmarks_celeba.txt", sep=r"\s+", skiprows=1)
        return dict(zip("celeba/img_celeba/" + a.index.astype(str), a.to_numpy(np.float32).reshape(-1, 5, 2)))
    if name == "vggface2_test":
        a = pd.read_csv(root / "vggface2_test/bb_landmark/loose_landmark_test.csv")
        return dict(zip("vggface2_test/test/" + a.iloc[:, 0] + ".jpg", a.iloc[:, 1:11].to_numpy(np.float32).reshape(-1, 5, 2)))
    return {}
