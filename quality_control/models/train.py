"""
OC Doors AI Quality Control — CV Feature Extraction + ML Training Pipeline
HOG + LBP + HSV + Edge + Gradient → Gradient Boosting Ensemble
Program Manager: OC Doors (formerly Masonite International)
"""
import cv2
import numpy as np
import os
import json
import time
import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

CLASSES  = ["good","crack","blister","crooked_corner",
            "thin_paint","thick_paint","scratch","delamination"]
IMG_SIZE = (224, 224)


def extract_features(img_input):
    """301-dimensional CV feature vector from door skin image."""
    img = cv2.imread(img_input) if isinstance(img_input, str) else img_input.copy()
    if img is None:
        return None
    img  = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feat = []

    # HOG — shape/gradient orientation (cracks, scratches, corners)
    hog = cv2.HOGDescriptor((64,64),(16,16),(8,8),(8,8),9)
    feat.extend(hog.compute(cv2.resize(gray,(64,64))).flatten()[::4][:200].tolist())

    # HSV histograms — paint thickness colour shifts
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for ch, bins in zip(range(3), [18, 8, 8]):
        h = cv2.calcHist([hsv], [ch], None, [bins], [0, 256])
        feat.extend(cv2.normalize(h, h).flatten().tolist())

    # LBP — micro-texture (blister bumps, orange-peel, delamination)
    lbp = np.zeros_like(gray, dtype=np.float32)
    for ai in range(24):
        a = 2 * np.pi * ai / 24
        sh = np.roll(np.roll(gray.astype(np.float32),
             int(round(-3*np.sin(a))),0), int(round(3*np.cos(a))),1)
        lbp += (sh >= gray.astype(np.float32)).astype(np.float32) * (2**ai)
    lh, _ = np.histogram(lbp.flatten(), bins=32, range=(0, 2**24))
    feat.extend((lh / max(lh.sum(), 1e-7)).tolist())

    # Canny edge stats — cracks and scratches produce many sharp edges
    edges = cv2.Canny(gray, 50, 150)
    feat.extend([float(edges.mean()), float(edges.std()),
                 float((edges > 0).sum()) / (224 * 224)])

    # Sobel gradient magnitude
    gx  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    feat.extend([float(mag.mean()), float(mag.std()),
                 float(np.percentile(mag, 90)),
                 float(np.percentile(mag, 99))])

    # Global + 9-zone regional intensity
    feat.extend([float(gray.mean()), float(gray.std()),
                 float(np.percentile(gray, 10)),
                 float(np.percentile(gray, 90))])
    h3, w3 = 224 // 3, 224 // 3
    for r in range(3):
        for c in range(3):
            z = gray[r*h3:(r+1)*h3, c*w3:(c+1)*w3]
            feat.extend([float(z.mean()), float(z.std())])

    # Contour geometry — delamination produces large irregular contours
    _, thr = cv2.threshold(edges, 10, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = sorted([cv2.contourArea(c) for c in cnts], reverse=True)[:5]
    while len(areas) < 5:
        areas.append(0)
    feat.extend(areas)
    feat.append(float(len(cnts)))

    return np.array(feat, dtype=np.float32)


def load_dataset(image_dir="data/images"):
    X_tr, y_tr, X_vl, y_vl = [], [], [], []
    with open(f"{image_dir}/manifest.json") as f:
        manifest = json.load(f)
    print(f"  Loading {manifest['total']} door skin images...")
    for item in manifest["manifest"]:
        feat = extract_features(item["path"])
        if feat is None:
            continue
        if item["split"] == "train":
            X_tr.append(feat); y_tr.append(item["label"])
        else:
            X_vl.append(feat); y_vl.append(item["label"])
    print(f"  Train: {len(X_tr)}  Val: {len(X_vl)}")
    return (np.array(X_tr), np.array(y_tr),
            np.array(X_vl), np.array(y_vl))


def train_all(X_tr, y_tr, X_vl, y_vl):
    scaler    = StandardScaler()
    X_tr_s    = scaler.fit_transform(X_tr)
    X_vl_s    = scaler.transform(X_vl)

    models_cfg = {
        "GradientBoosting": (
            GradientBoostingClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.08,
                subsample=0.85, random_state=42),
            False),
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=200, min_samples_leaf=2,
                random_state=42, n_jobs=-1),
            False),
        "SVM_RBF": (
            SVC(C=5.0, gamma="scale", kernel="rbf",
                probability=True, random_state=42),
            True),
    }

    results = {}; trained = {}
    for name, (model, use_s) in models_cfg.items():
        t0  = time.time()
        Xtr = X_tr_s if use_s else X_tr
        Xvl = X_vl_s if use_s else X_vl
        print(f"  Training {name}...")
        model.fit(Xtr, y_tr)
        preds = model.predict(Xvl)
        elapsed = round(time.time() - t0, 1)

        rep = classification_report(y_vl, preds, output_dict=True, zero_division=0)
        cm  = confusion_matrix(y_vl, preds, labels=CLASSES)

        results[name] = {
            "accuracy":       round(accuracy_score(y_vl, preds), 4),
            "f1_weighted":    round(f1_score(y_vl, preds, average="weighted"), 4),
            "f1_macro":       round(f1_score(y_vl, preds, average="macro"), 4),
            "precision_w":    round(precision_score(y_vl, preds, average="weighted", zero_division=0), 4),
            "recall_w":       round(recall_score(y_vl, preds, average="weighted", zero_division=0), 4),
            "train_time_s":   elapsed,
            "per_class":      {c: round(rep.get(c,{}).get("f1-score",0),3)  for c in CLASSES},
            "per_class_f1":   {c: round(rep.get(c,{}).get("f1-score",0),3)  for c in CLASSES},
            "per_class_prec": {c: round(rep.get(c,{}).get("precision",0),3) for c in CLASSES},
            "per_class_rec":  {c: round(rep.get(c,{}).get("recall",0),3)    for c in CLASSES},
            "confusion_matrix": cm.tolist(),
        }
        trained[name] = (model, use_s)
        m = results[name]
        print(f"    Acc={m['accuracy']:.1%}  F1={m['f1_weighted']:.4f}  [{elapsed}s]")

    best = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\n  Best: {best} ({results[best]['accuracy']:.1%})")
    return trained, results, best, scaler


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    if not os.path.exists("data/images/manifest.json"):
        from data.generate_images import generate_dataset
        generate_dataset()

    print("Extracting features from door skin images...")
    X_tr, y_tr, X_vl, y_vl = load_dataset()

    print("\nTraining CV classifiers...")
    trained, results, best, scaler = train_all(X_tr, y_tr, X_vl, y_vl)

    best_model, _ = trained[best]
    joblib.dump(best_model, "models/qc_model.pkl")
    joblib.dump(scaler,     "models/scaler.pkl")

    with open("models/model_results.json", "w") as f:
        json.dump({"best_model": best, "results": results,
                   "classes": CLASSES}, f, indent=2)

    from collections import Counter
    with open("models/class_distribution.json", "w") as f:
        json.dump(dict(Counter(y_tr)), f, indent=2)

    print("\n✅ OC Doors QC model saved")
    for n, m in results.items():
        star = "  ← BEST" if n == best else ""
        print(f"  {n:<20} Acc={m['accuracy']:.1%}  F1={m['f1_weighted']:.4f}{star}")
    print(f"\nPer-class F1 ({best}):")
    for cls, f1 in results[best]["per_class"].items():
        sev = ["None","Minor","Moderate","Major"][
            {"good":0,"thin_paint":1,"thick_paint":1,"scratch":2,
             "blister":2,"crooked_corner":3,"crack":3,"delamination":3}[cls]]
        print(f"  {cls:<20} {f1:.3f}  [{sev}]")
