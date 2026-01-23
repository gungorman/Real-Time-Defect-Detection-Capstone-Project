import argparse
import os
import random
import shutil
from pathlib import Path
from collections import defaultdict, Counter

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(images_dir: Path):
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    imgs = []
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
    return sorted(imgs)


def read_label_classes(label_path: Path):
    """
    YOLO label format per line:
      class x_center y_center width height
    Returns a list of class ints found in file (may be empty).
    """
    classes = []
    if not label_path.exists():
        return classes
    txt = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return classes
    for line in txt.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
            classes.append(cls)
        except Exception:
            # Skip malformed lines
            continue
    return classes


def build_image_records(dataset_root: Path):
    """
    Expects:
      dataset_root/images
      dataset_root/labels
    Returns list of dict:
      {
        "img": Path,
        "label": Path,
        "classes": [int...],
        "main_class": int or None
      }
    """
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    imgs = list_images(images_dir)
    records = []

    missing_labels = 0
    for img_path in imgs:
        label_path = labels_dir / (img_path.stem + ".txt")
        classes = read_label_classes(label_path)
        if not label_path.exists():
            missing_labels += 1

        # Main class heuristic:
        # - If multiple classes, choose the smallest id (stable).
        # - If no labels, main_class=None (goes into a separate pool).
        main_class = min(classes) if classes else None

        records.append(
            {
                "img": img_path,
                "label": label_path,
                "classes": classes,
                "main_class": main_class,
            }
        )

    if missing_labels:
        print(f"[WARN] {missing_labels} images in {dataset_root} have no matching label .txt")

    return records


def sample_without_replacement(records, n, rng: random.Random):
    if n > len(records):
        raise ValueError(f"Requested {n} samples but only {len(records)} available.")
    chosen = rng.sample(records, n)
    chosen_set = {id(x): x for x in chosen}  # stable identity map
    remaining = [r for r in records if id(r) not in chosen_set]
    return chosen, remaining


def balanced_sample_from_train(records, n, rng: random.Random):
    """
    Attempts to pick n records from TRAIN with even distribution across classes.
    Uses each image's 'main_class' for balancing.
    Images with main_class=None are excluded from balancing and only used as fallback.

    Strategy:
    - Group by main_class (excluding None).
    - Compute base per-class quota = n // num_classes.
    - Distribute remainder round-robin across classes with available items.
    - If any class runs out, backfill from remaining pool (including None-class pool).
    """
    # Group by main class
    by_class = defaultdict(list)
    none_pool = []
    for r in records:
        if r["main_class"] is None:
            none_pool.append(r)
        else:
            by_class[r["main_class"]].append(r)

    class_ids = sorted(by_class.keys())
    if not class_ids:
        # No labeled classes at all -> just random sample
        return sample_without_replacement(records, n, rng)

    # Shuffle each bucket for randomness
    for cid in class_ids:
        rng.shuffle(by_class[cid])
    rng.shuffle(none_pool)

    num_classes = len(class_ids)
    base = n // num_classes
    remainder = n % num_classes

    chosen = []
    used_ids = set()

    # Take base quota from each class
    for cid in class_ids:
        take = min(base, len(by_class[cid]))
        chosen.extend(by_class[cid][:take])
        used_ids.update(id(x) for x in by_class[cid][:take])
        by_class[cid] = by_class[cid][take:]

    # Distribute remainder round-robin
    rr = class_ids[:]
    rng.shuffle(rr)
    for _ in range(remainder):
        picked = False
        for cid in rr:
            if by_class[cid]:
                x = by_class[cid].pop()
                chosen.append(x)
                used_ids.add(id(x))
                picked = True
                break
        if not picked:
            break

    # If still short, backfill from any remaining labeled + none_pool
    if len(chosen) < n:
        remaining_pool = []
        for cid in class_ids:
            remaining_pool.extend(by_class[cid])
        remaining_pool.extend(none_pool)
        rng.shuffle(remaining_pool)

        need = n - len(chosen)
        for x in remaining_pool:
            if id(x) in used_ids:
                continue
            chosen.append(x)
            used_ids.add(id(x))
            need -= 1
            if need == 0:
                break

    if len(chosen) < n:
        raise ValueError(
            f"Could not collect {n} balanced samples. Only got {len(chosen)} from training pool."
        )

    remaining = [r for r in records if id(r) not in used_ids]
    return chosen, remaining


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def copy_records(records, out_split_dir: Path):
    out_images = out_split_dir / "images"
    out_labels = out_split_dir / "labels"
    ensure_dir(out_images)
    ensure_dir(out_labels)

    copied = 0
    missing_label = 0

    for r in records:
        shutil.copy2(r["img"], out_images / r["img"].name)
        if r["label"].exists():
            shutil.copy2(r["label"], out_labels / r["label"].name)
        else:
            missing_label += 1
            # Still create an empty label file so YOLO doesn't crash
            (out_labels / (r["img"].stem + ".txt")).write_text("", encoding="utf-8")
        copied += 1

    return copied, missing_label


def summarize(records, name):
    main_classes = [r["main_class"] for r in records if r["main_class"] is not None]
    c = Counter(main_classes)
    return f"{name}: {len(records)} images | main_class distribution: {dict(sorted(c.items()))}"


def main():
    # ---- HARD-CODE YOUR PATHS HERE ----
    train_root = Path(r"data/train_data")   # original training dataset root
    test_root  = Path(r"data/test_data")    # original test dataset root
    out_root   = Path(r"data")              # output dataset root

    seed = 42
    rng = random.Random(seed)

    train_records = build_image_records(train_root)
    test_records = build_image_records(test_root)

    # ---- Sample from original test (no class balancing requested there) ----
    new_test_from_test, test_records = sample_without_replacement(test_records, 100, rng)
    new_val_from_test, test_records = sample_without_replacement(test_records, 50, rng)
    new_train_from_test, test_records = sample_without_replacement(test_records, 50, rng)

    # ---- Sample from original train (balanced) ----
    new_test_from_train, train_records = balanced_sample_from_train(train_records, 100, rng)
    new_val_from_train, train_records = balanced_sample_from_train(train_records, 200, rng)
    new_train_from_train, train_records = balanced_sample_from_train(train_records, 1050, rng)

    new_test = new_test_from_test + new_test_from_train
    new_val = new_val_from_test + new_val_from_train
    new_train = new_train_from_test + new_train_from_train

    # ---- Write output ----
    ensure_dir(out_root)

    test_out = out_root / "test"
    val_out = out_root / "val"
    train_out = out_root / "train"

    copied_t, miss_t = copy_records(new_test, test_out)
    copied_v, miss_v = copy_records(new_val, val_out)
    copied_tr, miss_tr = copy_records(new_train, train_out)

    # ---- Print summary ----
    print("=== Split summary ===")
    print(summarize(new_test, "NEW TEST"))
    print(summarize(new_val, "NEW VAL"))
    print(summarize(new_train, "NEW TRAIN"))
    print()
    print("=== Copy summary ===")
    print(f"test:  copied={copied_t}, missing_labels_created_empty={miss_t}")
    print(f"val:   copied={copied_v}, missing_labels_created_empty={miss_v}")
    print(f"train: copied={copied_tr}, missing_labels_created_empty={miss_tr}")
    print()
    print(f"Output written to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
