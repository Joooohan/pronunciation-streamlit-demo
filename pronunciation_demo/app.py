import json
import logging
import os
import sys
from io import BytesIO

import noisereduce as nr
import numpy as np
import requests
import s3fs
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from librosa import load
from PIL import Image

sys.path.append(os.path.dirname(__file__))

from phoneme_recognition.models import ModelWrapper
from phoneme_recognition.phonology import convert, transcribe

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@st.cache(hash_funcs={ModelWrapper: lambda _: None})
def load_model():
    """Load pretrained model from S3 bucket."""
    MODEL_NAME = "wav2vec2-base-timit-phonemes-15e"
    LOCAL_MODEL_URI = os.path.expanduser(
        os.path.join(
            "~",
            ".cache",
            "streamlit-prononciation-demo",
            MODEL_NAME,
        )
    )
    S3_MODEL_URI = f"s3://streamlit-pronunciation-demo/{MODEL_NAME}/"

    if not os.path.exists(LOCAL_MODEL_URI):
        s3 = s3fs.S3FileSystem()
        logger.info("Downloading model weights...")
        s3.download(S3_MODEL_URI, LOCAL_MODEL_URI, recursive=True)
        logger.info("Finished downloading.")
    else:
        logger.info("Found existing weights.")

    return ModelWrapper(LOCAL_MODEL_URI)


@st.cache
def load_examples():
    logger.info("Loading examples")
    dirpath = os.path.dirname(__file__)
    with open(os.path.join(dirpath, "examples.json")) as f:
        examples = json.load(f)
    return examples


model = load_model()
examples = load_examples()

# Stateful app storing the example index
if "idx" not in st.session_state:
    st.session_state["idx"] = 0


def next_idx():
    st.session_state["idx"] = (st.session_state["idx"] + 1) % len(examples)


# Document
st.title("Deep learning for phoneme recognition")

st.write(
    """
    This app is a demonstration of a simple deep learning model tuned for
    for phoneme recognition.

    You can read the detailed article about the design and training here.

    This app's code is available [here](https://github.com/Joooohan/pronunciation-streamlit-demo)
"""
)

st.subheader("Demo")

st.write(
    """
    Try to pronounce the following word in the correct way. The example's
    standard pronunciation is shown below the picture. The pronunciation is
    described using the Wikipedia respelling method. It is pretty intuitive but
    you can find more information [here](https://en.wikipedia.org/wiki/Pronunciation_respelling_for_English).
    """
)

# Load example
example = examples[st.session_state["idx"]]
url = example["imageSrc"]
resp = requests.get(url)
resp.raise_for_status()
img = np.array(Image.open(BytesIO(resp.content)).resize((450, 300)))
st.image(img, caption=example["word"])
standard_pronunciation, _ = transcribe(example["word"])
standard_pronunciation = convert(standard_pronunciation, "timit", "wikipedia")
audio_bytes = audio_recorder()
st.write(f"Standard pronunciation: {' '.join(standard_pronunciation)}")

# Record & analyze audio
if audio_bytes:
    speech, sampling_rate = load(BytesIO(audio_bytes), sr=16000)
    noiseless = nr.reduce_noise(y=speech, sr=sampling_rate, y_noise=speech[:8000])
    pred = model.predict(noiseless, sampling_rate).split(" ")
    pred = convert(pred, "timit", "wikipedia")
    st.write(f"Inferred pronunciation: {' '.join(pred)}")
    st.audio(audio_bytes, format="audio/wav")

st.button(label="Next example", on_click=next_idx)
