import os
import time
import base64
from io import BytesIO
from PIL import Image

from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
from lmdeploy.vl import load_image                     # ← make sure this line is present
from lmdeploy.vl.constants import IMAGE_TOKEN

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix
)
import pandas as pd

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
MODEL_NAME  = "OpenGVLab/InternVL3-14B"
pipe = pipeline(
    MODEL_NAME,
    backend_config=TurbomindEngineConfig(session_len=32000, tp=1),
    chat_template_config=ChatTemplateConfig(model_name="internvl2_5")
)

DATA_ROOT   = "./data"
OUTPUT_DIR  = "./result"
SHOT        = 1
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def build_prompt(class_labels, shot):
    lines = []
    if shot > 0:
        for idx, cls in enumerate(class_labels, start=1):
            lines.append(f"Image-{idx}: {IMAGE_TOKEN}")
            lines.append(f"This image is class: {cls}")
        test_idx = len(class_labels) + 1
        lines.append(f"Image-{test_idx}: {IMAGE_TOKEN}")
        lines.append(
            f"Classify Image-{test_idx}. Choices: {', '.join(class_labels)}. "
            "Reply with exactly one class name (1–2 words)."
        )
    else:
        lines.append(f"Image-1: {IMAGE_TOKEN}")
        lines.append(
            f"Classify Image-1 based on your knowledge alone. "
            f"Choices: {', '.join(class_labels)}. "
            "Reply with exactly one class name (1–2 words)."
        )
    return "\n".join(lines)

def summarize_prompt(prompt):
    return prompt.replace(IMAGE_TOKEN, "[IMAGE]")

def extract_prediction(text, class_labels):
    txt = text.lower()
    for cls in sorted(class_labels, key=len, reverse=True):
        if cls.lower() in txt:
            return cls
    parts = text.strip().split()
    return parts[0] if parts else ""

def pil_to_data_url(img: Image.Image):
    buf = BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

# ─── MAIN LOOP ──────────────────────────────────────────────────────────────────
for dataset in os.listdir(DATA_ROOT):
    ds_path = os.path.join(DATA_ROOT, dataset)
    out_csv = os.path.join(OUTPUT_DIR, f"metrics_{dataset}_internvl.csv")
    if not os.path.isdir(ds_path) or os.path.exists(out_csv):
        continue

    print(f">>> Evaluating {dataset}")

    # 1) Load reference images
    ref_dir      = os.path.join(ds_path, "REFERENCE")
    class_labels = sorted(os.listdir(ref_dir))
    ref_pils     = []
    for cls in class_labels:
        cls_folder = os.path.join(ref_dir, cls)
        fname = next(
            f for f in os.listdir(cls_folder)
            if os.path.isfile(os.path.join(cls_folder, f))
        )
        ref_pils.append(
            Image.open(os.path.join(cls_folder, fname)).convert("RGB")
        )

    prompt_template = build_prompt(class_labels, SHOT)
    y_true, y_pred  = [], []

    # 2) Classify each TEST image
    for cls in class_labels:
        test_folder = os.path.join(ds_path, "TEST", cls)
        for fname in os.listdir(test_folder):
            img_path = os.path.join(test_folder, fname)
            print(f"\nClassifying: {dataset}/{cls}/{fname}")

            orig_test = Image.open(img_path).convert("RGB")
            scale     = 1.0
            response  = ""

            # 3) Downscale refs+test until non-empty
            for _ in range(5):
                # resize refs & test
                scaled_refs = [
                    im.resize(
                        (int(im.width*scale), int(im.height*scale)),
                        Image.LANCZOS
                    )
                    for im in ref_pils
                ]
                scaled_test = orig_test.resize(
                    (int(orig_test.width*scale),
                     int(orig_test.height*scale)),
                    Image.LANCZOS
                )

                # convert to data-URLs and load
                imgs = scaled_refs + [scaled_test] if SHOT>0 else [scaled_test]
                images = [load_image(pil_to_data_url(im)) for im in imgs]

                print(f"--- Prompt (scale={scale:.2f}) ---")
                print(summarize_prompt(prompt_template))
                print("--- End prompt ---")

                resp = pipe((prompt_template, images))
                out  = resp.text.strip()
                if out:
                    response = out
                    break

                scale /= 2
                print("⚠️  Empty response; downscaling further…")

            if not response:
                print("❌ Still empty; proceeding with blank output.")

            pred    = extract_prediction(response, class_labels)
            correct = (pred == cls)
            print(f"Model output: '{response}'")
            print(f"Predicted: {pred} | True: {cls} | "
                  f"{'✅' if correct else '❌'}")

            y_true.append(cls)
            y_pred.append(pred)
            time.sleep(0.1)

    # 4) Compute & save metrics
    report = classification_report(
        y_true, y_pred, labels=class_labels,
        output_dict=True, zero_division=0
    )
    df = pd.DataFrame(report).T
    df.loc["accuracy", :] = [
        accuracy_score(y_true, y_pred), None, None, len(y_true)
    ]
    kappa = cohen_kappa_score(y_true, y_pred)
    cm    = confusion_matrix(y_true, y_pred, labels=class_labels)
    n     = len(y_true)
    tp, fp, fn, tn = {}, {}, {}, {}
    for i, c in enumerate(class_labels):
        tp[c] = int(cm[i, i])
        fp[c] = int(cm[:, i].sum() - cm[i, i])
        fn[c] = int(cm[i, :].sum() - cm[i, i])
        tn[c] = n - tp[c] - fp[c] - fn[c]
    df["tp"], df["fp"], df["fn"], df["tn"] = (
        pd.Series(tp), pd.Series(fp),
        pd.Series(fn), pd.Series(tn)
    )
    df.loc["cohen_kappa", [
        "precision","recall","f1-score","support","tp","fp","fn","tn"
    ]] = [None, None, kappa, None, None, None, None, None]

    df.to_csv(out_csv, index=True)
    print(f"\n→ Saved metrics to {out_csv}\n")
