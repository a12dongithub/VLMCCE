import os
import time
from PIL import Image
import torch
from llava.model.builder import load_pretrained_model
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN
)
from llava.mm_utils import tokenizer_image_token, process_images
from llava.conversation import conv_templates
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix
)
import pandas as pd

# REQUIRES THIS SCRIPT TO BE INSIDE THE LLAVA MED OFFICIAL REPO CLONE

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = "microsoft/llava-med-v1.5-mistral-7b"
CONV_MODE   = "vicuna_v1"
DATA_ROOT   = r"C:\Users\samar\Documents\CVPR\Data"      # root folder with Dataset1, Dataset2, …
OUTPUT_DIR  = "./"      # where CSVs go
SHOT        = 1                # 0 = zero-shot, >0 = one-shot
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── LOAD MODEL ─────────────────────────────────────────────────────────────────
tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path=MODEL_PATH,
    model_base=None,
    model_name=MODEL_PATH
)
# ensure pad token
pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
model.config.pad_token_id = pad_id

# prepare conversation template
conv_template = conv_templates[CONV_MODE]

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def build_shot_prompt(class_labels):
    lines = []
    if SHOT > 0:
        lines.append("Here are reference images—one shot per class:")
        for i, cls in enumerate(class_labels, start=1):
            lines.append(f"Ref-{i}: {DEFAULT_IMAGE_TOKEN}")
            lines.append(f"This image is class: {cls}")
        lines.append(
            f"Now classify the test image using reference images and your knowledge. "
            f"Choices: {', '.join(class_labels)}. "
            "Reply with the class name in one or two words maximum."
        )
    else:
        lines.append(
            f"Classify the test image based on your knowledge alone. "
            f"Choices: {', '.join(class_labels)}. "
            "Reply with the class name in one or two words maximum."
        )
    lines.append(f"Test: {DEFAULT_IMAGE_TOKEN}")
    return "\n".join(lines)

def summarize_prompt(txt):
    return txt.replace(DEFAULT_IMAGE_TOKEN, "[IMAGE]")

def extract_prediction(resp_text, class_labels):
    txt = resp_text.lower()
    for cls in sorted(class_labels, key=len, reverse=True):
        if cls.lower() in txt:
            return cls
    return resp_text.strip().split()[0] if resp_text else ""

# ─── MAIN EVALUATION LOOP ────────────────────────────────────────────────────────
for dataset in os.listdir(DATA_ROOT):
    ds_path = os.path.join(DATA_ROOT, dataset)
    if not os.path.isdir(ds_path): continue
    print(f"\n>>> Evaluating dataset: {dataset}")

    # load classes & reference images
    ref_dir      = os.path.join(ds_path, "REFERENCE")
    class_labels = sorted(os.listdir(ref_dir))
    ref_pils     = []
    for cls in class_labels:
        fp = next(f for f in os.listdir(os.path.join(ref_dir, cls))
                  if os.path.isfile(os.path.join(ref_dir, cls, f)))
        pil = Image.open(os.path.join(ref_dir, cls, fp)).convert("RGB")
        ref_pils.append(pil)

    # preprocess reference images into a batch tensor
    if SHOT > 0:
        ref_tensors = process_images(ref_pils, image_processor, model.config)  # Tensor [N, C, H, W]

    prompt_template = build_shot_prompt(class_labels)
    y_true, y_pred = [], []
    test_root = os.path.join(ds_path, "TEST")

    for cls in class_labels:
        cls_folder = os.path.join(test_root, cls)
        for fname in os.listdir(cls_folder):
            img_path = os.path.join(cls_folder, fname)
            print(f"Classifying: {dataset}/{cls}/{fname}")

            # load & preprocess test image
            pil_test = Image.open(img_path).convert("RGB")
            test_tensor = process_images([pil_test], image_processor, model.config)[0]

            # assemble image batch correctly
            if SHOT > 0:
                test_batch = test_tensor.unsqueeze(0)                   # [1, C, H, W]
                imgs = torch.cat([ref_tensors, test_batch], dim=0)      # [N+1, C, H, W]
            else:
                imgs = test_tensor.unsqueeze(0)                         # [1, C, H, W]

            # build & display prompt
            prompt = prompt_template
            print("--- Prompt ---")
            print(summarize_prompt(prompt))
            print("--- End Prompt ---")

            # conversation
            conv = conv_template.copy()
            roles = conv.roles
            conv.append_message(roles[0], prompt)
            conv.append_message(roles[1], None)
            full_prompt = conv.get_prompt()

            # tokenize with image token
            input_ids = tokenizer_image_token(
                full_prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt"
            ).unsqueeze(0).cuda()

            # generate
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=imgs.half().cuda(),
                    do_sample=False,
                    temperature=0.0,
                    max_new_tokens=256
                )
            resp_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print(resp_text)
            # extract & record
            pred    = extract_prediction(resp_text, class_labels)
            correct = (pred == cls)
            print(f"Model output: '{resp_text}'")
            print(f"Predicted: {pred} | True: {cls} | {'CORRECT' if correct else 'INCORRECT'}\n")

            y_true.append(cls)
            y_pred.append(pred)
            time.sleep(0.1)

    # compute metrics
    rpt = classification_report(y_true, y_pred,
                                labels=class_labels,
                                output_dict=True,
                                zero_division=0)
    df = pd.DataFrame(rpt).T
    df.loc["accuracy", :] = [accuracy_score(y_true, y_pred), None, None, len(y_true)]

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

    out_csv = os.path.join(OUTPUT_DIR, f"metrics_{dataset}_llavamed.csv")
    df.to_csv(out_csv, index=True)
    print(f"→ Saved metrics to {out_csv}")
