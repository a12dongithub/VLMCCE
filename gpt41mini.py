import os
import time
import base64
import openai
from openai import OpenAI
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix
)
import pandas as pd

# ─── CONFIG ─────────────────────────────────────────────────────────
API_KEY     = ""               # ← your actual API key                   # set your key here
MODEL_NAME  = "gpt-4.1"            # swap in gpt-4.1-mini, gpt-4.1, etc.
DATA_ROOT   = "./Data"
OUTPUT_DIR  = "./results0"
SHOT        = 0                    # 0 = zero-shot; >0 = one-shot
os.makedirs(OUTPUT_DIR, exist_ok=True)

client = OpenAI(api_key=API_KEY)

# ─── HELPERS ───────────────────────────────────────────────────────
def encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_vision_content(ref_map, test_b64, class_labels):
    content = []
    if SHOT > 0:
        content.append({
            "type": "input_text",
            "text": "Here are reference images—one shot per class:"
        })
        for label, b64 in ref_map.items():
            content.extend([
                {
                    "type":      "input_image",
                    "image_url": f"data:image/png;base64,{b64}",
                    "detail":    "auto"
                },
                {
                    "type": "input_text",
                    "text": f"This image is class: {label}"
                }
            ])
        prompt_text = (
            f"Now classify this test image using reference images and your knowledge. "
            f"Choices: {', '.join(class_labels)}. "
            "Reply with the class name in one or two words maximum."
        )
    else:
        prompt_text = (
            f"Classify this test image based on your knowledge alone. "
            f"Choices: {', '.join(class_labels)}. "
            "Reply with the class name in one or two words maximum."
        )

    content.append({"type": "input_text", "text": prompt_text})
    content.append({
        "type":      "input_image",
        "image_url": f"data:image/png;base64,{test_b64}",
        "detail":    "auto"
    })
    return content

def summarize_prompt(content):
    lines = []
    for block in content:
        if block["type"] == "input_text":
            lines.append(block["text"])
        else:
            lines.append("[IMAGE]")
    return "\n".join(lines)

def extract_prediction(full_text, class_labels):
    txt = full_text.lower()
    for cls in sorted(class_labels, key=len, reverse=True):
        if cls.lower() in txt:
            return cls
    return full_text.strip().split()[0]

# ─── MAIN LOOP ───────────────────────────────────────────────────────
for dataset in os.listdir(DATA_ROOT):
    ds_path = os.path.join(DATA_ROOT, dataset)
    if not os.path.isdir(ds_path):
        continue

    print(f">>> Evaluating dataset: {dataset}")

    # 1) load references
    ref_dir      = os.path.join(ds_path, "REFERENCE")
    class_labels = sorted(os.listdir(ref_dir))
    ref_map = {}
    for cls in class_labels:
        img_file = next(f for f in os.listdir(os.path.join(ref_dir, cls))
                        if os.path.isfile(os.path.join(ref_dir, cls, f)))
        ref_map[cls] = encode_image_b64(os.path.join(ref_dir, cls, img_file))

    y_true, y_pred = [], []

    # 2) test loop
    for cls in class_labels:
        test_cls_dir = os.path.join(ds_path, "TEST", cls)
        for img_file in os.listdir(test_cls_dir):
            img_path = os.path.join(test_cls_dir, img_file)
            test_b64 = encode_image_b64(img_path)

            content = build_vision_content(ref_map, test_b64, class_labels)
            print("\n--- Prompt being sent ---")
            print(summarize_prompt(content))
            print("--- End prompt ---\n")

            # retry loop
            while True:
                try:
                    resp = client.responses.create(
                        model = MODEL_NAME,
                        input = [{"role":"user","content":content}]
                    )
                    break
                except openai.error.RateLimitError:
                    print("429 rate-limit hit; retrying in 1s…")
                    time.sleep(1)

            full_out = resp.output_text.strip()
            pred     = extract_prediction(full_out, class_labels)
            correct  = (pred == cls)

            print(f"Model output: '{full_out}'")
            print(f"Predicted: {pred} | True: {cls} | {'CORRECT' if correct else 'INCORRECT'}\n")

            y_true.append(cls)
            y_pred.append(pred)
            time.sleep(0.1)

    # 3) metrics
    report = classification_report(
        y_true, y_pred,
        labels=class_labels,
        output_dict=True,
        zero_division=0
    )
    df = pd.DataFrame(report).T
    df.loc["accuracy", :] = [
        accuracy_score(y_true, y_pred), None, None, len(y_true)
    ]

    kappa = cohen_kappa_score(y_true, y_pred)
    cm    = confusion_matrix(y_true, y_pred, labels=class_labels)
    n     = len(y_true)
    tp, fp, fn, tn = {}, {}, {}, {}
    for i, cls in enumerate(class_labels):
        tp[cls] = int(cm[i, i])
        fp[cls] = int(cm[:, i].sum() - cm[i, i])
        fn[cls] = int(cm[i, :].sum() - cm[i, i])
        tn[cls] = n - tp[cls] - fp[cls] - fn[cls]
    df["tp"], df["fp"], df["fn"], df["tn"] = (
        pd.Series(tp), pd.Series(fp), pd.Series(fn), pd.Series(tn)
    )
    df.loc["cohen_kappa", ["precision","recall","f1-score","support","tp","fp","fn","tn"]] = (
        [None, None, kappa, None, None, None, None, None]
    )

    out_csv = os.path.join(OUTPUT_DIR, f"metrics_{dataset}_{MODEL_NAME}.csv")
    df.to_csv(out_csv)
    print(f"→ Saved metrics to {out_csv}\n")
