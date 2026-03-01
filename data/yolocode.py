"""
End-to-end YOLO training + Kaggle-style submission from:
- /mnt/data/dataset/data/training_images
- /mnt/data/dataset/data/testing_images
- /mnt/data/dataset/data/train_solution_bounding_boxes (1).csv
- /mnt/data/dataset/data/sample_submission.csv

What it does:
1) Converts CSV bounding boxes -> YOLO label txt files
2) Splits train/val and builds YOLO folder structure
3) Trains YOLOv8 (ultralytics)
4) Runs inference on test images
5) Writes submission.csv with normalized xyxy boxes flattened as a string

Run:
pip install ultralytics opencv-python pillow scikit-learn pandas pyyaml
python train_yolo_and_submit.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split

from ultralytics import YOLO



from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


ROOT = BASE_DIR

TRAIN_IMAGES_DIR = ROOT / "training_images"
TEST_IMAGES_DIR  = ROOT / "testing_images"

TRAIN_CSV  = ROOT / "train_solution_bounding_boxes (1).csv"
SAMPLE_SUB = ROOT / "sample_submission.csv"


OUT = ROOT / "yolo_work"
YOLO_DS = OUT / "yolo_dataset"
SUBMISSION_PATH = OUT / "submission.csv"


def ensure_clean_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def img_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def xyxy_to_yolo_norm(xmin, ymin, xmax, ymax, w, h):

    xmin = max(0.0, min(float(xmin), w - 1.0))
    ymin = max(0.0, min(float(ymin), h - 1.0))
    xmax = max(0.0, min(float(xmax), w - 1.0))
    ymax = max(0.0, min(float(ymax), h - 1.0))

    bw = max(0.0, xmax - xmin)
    bh = max(0.0, ymax - ymin)
    xc = xmin + bw / 2.0
    yc = ymin + bh / 2.0


    return xc / w, yc / h, bw / w, bh / h


def write_yolo_label(label_path: Path, rows: pd.DataFrame, w: int, h: int, class_id: int = 0) -> None:

    lines = []
    for r in rows.itertuples(index=False):
        xc, yc, bw, bh = xyxy_to_yolo_norm(r.xmin, r.ymin, r.xmax, r.ymax, w, h)

        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def build_data_yaml(path: Path, train_images: str, val_images: str) -> None:
    data = {
        "path": str(YOLO_DS),
        "train": train_images,
        "val": val_images,
        "names": {0: "object"},
        "nc": 1,
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False))



def prepare_yolo_dataset(val_size: float = 0.2, seed: int = 42) -> Path:
    ensure_clean_dir(OUT)
    ensure_clean_dir(YOLO_DS)

    (YOLO_DS / "images/train").mkdir(parents=True, exist_ok=True)
    (YOLO_DS / "images/val").mkdir(parents=True, exist_ok=True)
    (YOLO_DS / "labels/train").mkdir(parents=True, exist_ok=True)
    (YOLO_DS / "labels/val").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TRAIN_CSV)
    all_imgs = sorted(df["image"].unique().tolist())

    train_imgs, val_imgs = train_test_split(all_imgs, test_size=val_size, random_state=seed)

    g = df.groupby("image", sort=False)

    def process_split(img_list, split: str):
        for img_name in img_list:
            src_img = TRAIN_IMAGES_DIR / img_name
            if not src_img.exists():
                raise FileNotFoundError(f"Missing image: {src_img}")


            dst_img = YOLO_DS / f"images/{split}/{img_name}"
            shutil.copy2(src_img, dst_img)


            w, h = img_size(src_img)
            rows = g.get_group(img_name) if img_name in g.groups else df.iloc[0:0]
            label_path = YOLO_DS / f"labels/{split}/{Path(img_name).stem}.txt"
            write_yolo_label(label_path, rows, w, h, class_id=0)

    process_split(train_imgs, "train")
    process_split(val_imgs, "val")

    data_yaml = OUT / "data.yaml"
    build_data_yaml(data_yaml, "images/train", "images/val")
    return data_yaml



def train_yolo(data_yaml: Path, epochs: int = 30, imgsz: int = 640):
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        device=0 if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else None,
        project=str(OUT / "runs"),
        name="exp",
        exist_ok=True,
    )


    best = OUT / "runs" / "exp" / "weights" / "best.pt"
    if not best.exists():

        best = OUT / "runs" / "exp" / "weights" / "last.pt"
    return best



def make_submission(weights_path: Path, conf: float = 0.25, iou: float = 0.45) -> None:
    model = YOLO(str(weights_path))

    sub = pd.read_csv(SAMPLE_SUB)
    out_rows = []

    for img_name in sub["image"].tolist():
        img_path = TEST_IMAGES_DIR / img_name
        if not img_path.exists():
            raise FileNotFoundError(f"Missing test image: {img_path}")

        w, h = img_size(img_path)

        results = model.predict(
            source=str(img_path),
            conf=conf,
            iou=iou,
            verbose=False,
        )[0]

        bounds_list = []
        if results.boxes is not None and len(results.boxes) > 0:
            xyxy = results.boxes.xyxy.cpu().numpy()
            for (xmin, ymin, xmax, ymax) in xyxy:
                xmin_n = max(0.0, min(float(xmin) / w, 1.0))
                ymin_n = max(0.0, min(float(ymin) / h, 1.0))
                xmax_n = max(0.0, min(float(xmax) / w, 1.0))
                ymax_n = max(0.0, min(float(ymax) / h, 1.0))
                bounds_list.extend([xmin_n, ymin_n, xmax_n, ymax_n])


        if not bounds_list:
            bounds_str = "0.0 0.0 1.0 1.0"
        else:
            bounds_str = " ".join(f"{v:.6f}" for v in bounds_list)

        out_rows.append({"image": img_name, "bounds": bounds_str})

    pd.DataFrame(out_rows).to_csv(SUBMISSION_PATH, index=False)
    print("Saved:", SUBMISSION_PATH)


def main():
    data_yaml = prepare_yolo_dataset(val_size=0.2, seed=42)
    weights = train_yolo(data_yaml, epochs=30, imgsz=640)
    make_submission(weights, conf=0.25, iou=0.45)


if __name__ == "__main__":
    main()