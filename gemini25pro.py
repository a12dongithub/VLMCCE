import os
import time
from google import genai
from google.genai import types, errors
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix
)
import pandas as pd

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
PROJECT_ID   = "" # Google cloud project ID
LOCATION     = "" # Server location
MODEL_NAME   = "gemini-2.5-pro-preview-03-25"
DATA_ROOT    = "./Data"
OUTPUT_DIR   = "./results0"
SHOT         = 0   # 0 = zero-shot, >0 = one-shot

os.makedirs(OUTPUT_DIR, exist_ok=True)

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)

config = types.GenerateContentConfig(
    temperature=0.0,
    max_output_tokens=1024,
    response_modalities=["TEXT"]
)

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def load_image_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def build_contents(ref_map, test_bytes, class_labels):
    parts = []
    if SHOT > 0:
        parts.append(types.Part(text="Here are reference images—one shot per class:"))
        for label, img_b in ref_map.items():
            parts.append(types.Part(inline_data=types.Blob(
                mime_type="image/png", data=img_b
            )))
            parts.append(types.Part(text=f"This image is class: {label}"))
        prompt = (
            f"Now classify this test image using reference images and your knowledge. "
            f"Choices: {', '.join(class_labels)}. "
            "Reply with the class name in one or two words maximum."
        )
    else:
        prompt = (
            f"Classify this test image based on your knowledge alone. "
            f"Choices: {', '.join(class_labels)}. "
            "Reply with the class name in one or two words maximum."
        )

    parts.append(types.Part(text=prompt))
    parts.append(types.Part(inline_data=types.Blob(
        mime_type="image/png", data=test_bytes
    )))
    return [types.Content(role="user", parts=parts)]

def summarize_contents(contents):
    lines = []
    for c in contents:
        for p in c.parts:
            lines.append(p.text if p.text is not None else "[IMAGE]")
    return "\n".join(lines)

def extract_prediction(full_text, class_labels):
    txt = (full_text or "empty").lower()
    for cls in sorted(class_labels, key=len, reverse=True):
        if cls.lower() in txt:
            return cls
    return full_text.strip().split()[0] if full_text.strip() else ""

def stream_with_retries(contents, max_retries=3):
    attempt, backoff = 0, 1.0
    while attempt < max_retries:
        try:
            return client.models.generate_content_stream(
                model=MODEL_NAME,
                contents=contents,
                config=config
            )
        except errors.ServerError as e:
            attempt += 1
            print(f"[Warning] ServerError (attempt {attempt}/{max_retries}): {e}. retrying in {backoff:.1f}s…")
            time.sleep(backoff)
            backoff *= 2
    # final try (will raise on error if it fails)
    return client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=contents,
        config=config
    )

# ─── MAIN LOOP ──────────────────────────────────────────────────────────────────
for dataset in os.listdir(DATA_ROOT):
    ds_path = os.path.join(DATA_ROOT, dataset)
    if not os.path.isdir(ds_path):
        continue

    out_csv = os.path.join(OUTPUT_DIR, f"metrics_{dataset}_gemini25pro.csv")
    if os.path.exists(out_csv):
        print(f"→ Skipping {dataset}, output already exists.")
        continue

    print(f">>> Evaluating dataset: {dataset}")
    # 1) Load reference images
    ref_dir      = os.path.join(ds_path, "REFERENCE")
    class_labels = sorted(os.listdir(ref_dir))
    ref_map = {}
    for cls in class_labels:
        cls_folder = os.path.join(ref_dir, cls)
        fn = next(f for f in os.listdir(cls_folder)
                  if os.path.isfile(os.path.join(cls_folder, f)))
        ref_map[cls] = load_image_bytes(os.path.join(cls_folder, fn))

    y_true, y_pred = [], []

    # 2) Iterate TEST images
    for cls in class_labels:
        test_folder = os.path.join(ds_path, "TEST", cls)
        for fn in os.listdir(test_folder):
            test_bytes = load_image_bytes(os.path.join(test_folder, fn))
            contents   = build_contents(ref_map, test_bytes, class_labels)

            print("\n--- Prompt being sent ---")
            print(summarize_contents(contents))
            print("--- End prompt ---\n")

            # turn the generator into a list, filter out None texts
            raw_chunks = stream_with_retries(contents)
            chunks = list(raw_chunks)
            texts  = [c.text for c in chunks if c.text is not None]
            full_text = "".join(texts).strip()

            pred    = extract_prediction(full_text, class_labels)
            correct = (pred == cls)
            print(f"Model output: '{full_text}'")
            print(f"Predicted: {pred} | True: {cls} | {'CORRECT' if correct else 'INCORRECT'}\n")

            y_true.append(cls)
            y_pred.append(pred)
            time.sleep(0.1)

    # 3) Compute metrics
    report = classification_report(
        y_true, y_pred,
        labels=class_labels,
        output_dict=True,
        zero_division=0
    )
    df = pd.DataFrame(report).T
    df.loc["accuracy", :] = [accuracy_score(y_true, y_pred), None, None, len(y_true)]

    kappa = cohen_kappa_score(y_true, y_pred)
    cm    = confusion_matrix(y_true, y_pred, labels=class_labels)
    n     = len(y_true)
    tp, fp, fn, tn = {}, {}, {}, {}
    for i, cls in enumerate(class_labels):
        tp[cls] = int(cm[i,i])
        fp[cls] = int(cm[:,i].sum() - cm[i,i])
        fn[cls] = int(cm[i,:].sum() - cm[i,i])
        tn[cls] = n - tp[cls] - fp[cls] - fn[cls]
    df["tp"], df["fp"], df["fn"], df["tn"] = (
        pd.Series(tp), pd.Series(fp),
        pd.Series(fn), pd.Series(tn)
    )
    df.loc["cohen_kappa", ["precision","recall","f1-score","support","tp","fp","fn","tn"]] = (
        [None, None, kappa, None, None, None, None, None]
    )

    # 4) Save CSV
    df.to_csv(out_csv, index=True)
    print(f"→ Saved metrics to {out_csv}\n")
