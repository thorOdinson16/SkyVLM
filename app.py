"""
Web UI: upload a satellite image, get a SkyVLM caption.

    streamlit run app.py

Uses checkpoints/vlm/clipvit_stage2/vlm_final.pt (override with SKYVLM_CKPT).
"""

import os
import sys
from pathlib import Path

import sentencepiece as spm
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from vlm import IMAGE_SIZE, MAX_CAPTION_TOKENS, TOKENIZER_PATH, build_vlm

CKPT = Path(os.environ.get("SKYVLM_CKPT", ROOT / "checkpoints" / "vlm" / "clipvit_stage2" / "vlm_final.pt"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREFIX = "a satellite image of "

TRANSFORM = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])


@st.cache_resource(show_spinner="Loading SkyVLM...")
def load_model():
    model = build_vlm()      # architecture; the encoder/LM weights are replaced by the checkpoint below
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=True)
    tok = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
    return model.to(DEVICE).eval(), tok


@torch.no_grad()
def caption(model, tok, image):
    x = TRANSFORM(image.convert("RGB"))[None].to(DEVICE)
    with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
        out = model.generate(x, tok.eos_id(), max_new_tokens=MAX_CAPTION_TOKENS)
    row = out[0].tolist()
    if tok.eos_id() in row:
        row = row[: row.index(tok.eos_id())]
    return tok.decode(row)


def parse(text):
    body = text[len(PREFIX):] if text.lower().startswith(PREFIX) else text
    main, _, rest = body.partition(", surrounded by ")
    return main.strip(), [o.strip() for o in rest.split(";") if o.strip()]


st.set_page_config(page_title="SkyVLM", page_icon="🛰️")
st.title("SkyVLM")
st.caption("Upload a satellite image to get a caption. 99M-parameter model trained on SkyScript.")

if not CKPT.exists():
    st.error(f"Checkpoint not found: {CKPT}")
    st.stop()

model, tok = load_model()

files = st.file_uploader("Satellite image(s)", type=["jpg", "jpeg", "png", "tif", "tiff", "webp"], accept_multiple_files=True)

for f in files:
    image = Image.open(f)
    left, right = st.columns([1, 1.4])
    left.image(image, caption=f.name, use_container_width=True)

    with right, st.spinner("Captioning..."):
        text = caption(model, tok, image)
        main, around = parse(text)

        st.markdown(f"**Main object:** {main}")
        if around:
            st.markdown("**Surrounding:**\n" + "\n".join(f"- {o}" for o in around))
        st.code(text, language=None)

    st.divider()

st.caption("Captions follow the SkyScript template (OpenStreetMap-style tags); the model does not take free-form prompts.")
