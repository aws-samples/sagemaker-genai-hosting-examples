"""Streamlit interface for the vLLM-Omni image-to-video workflow."""

from __future__ import annotations

import time
from pathlib import Path

import boto3
import streamlit as st

from vllm_omni_media import (
    DEFAULT_STATE_PATH,
    invoke_image,
    load_state,
    submit_video,
    validate_mp4,
    wait_for_s3_object,
)

st.set_page_config(page_title="vLLM-Omni image to video", layout="wide")
st.title("Generate images and video with vLLM-Omni")

state_path = Path(
    st.sidebar.text_input("Deployment state", str(DEFAULT_STATE_PATH))
).expanduser()

try:
    state = load_state(state_path)
except (FileNotFoundError, TypeError, ValueError) as error:
    st.info(str(error))
    st.stop()

session = boto3.Session(region_name=state.region)
runtime = session.client("sagemaker-runtime")
s3 = session.client("s3")

image_prompt = st.text_area(
    "Image prompt",
    (
        "Cinematic photograph of a small observatory on a windswept coastal "
        "cliff at sunrise, detailed clouds, natural light"
    ),
)
image_col, seed_col = st.columns([3, 1])
with image_col:
    image_size = st.selectbox("Image size", ["1024x1024", "768x768"])
with seed_col:
    seed = st.number_input("Seed", min_value=0, value=42, step=1)

if st.button("Generate image", type="primary", width="stretch"):
    with st.spinner("Generating image"):
        started = time.perf_counter()
        st.session_state.image_bytes = invoke_image(
            runtime,
            state,
            image_prompt,
            size=image_size,
            seed=int(seed),
        )
        st.session_state.image_seconds = time.perf_counter() - started
        st.session_state.pop("video_bytes", None)

if "image_bytes" in st.session_state:
    st.image(
        st.session_state.image_bytes,
        caption=f"FLUX.2-klein output in {st.session_state.image_seconds:.1f}s",
        width="stretch",
    )
    st.download_button(
        "Download image",
        st.session_state.image_bytes,
        "flux2-klein.png",
        "image/png",
        width="stretch",
    )

st.divider()
video_prompt = st.text_area(
    "Motion prompt",
    (
        "Slow camera push-in toward the coastal observatory as clouds drift "
        "across the sky and ocean waves move below; preserve the building, "
        "coastline, and composition"
    ),
)
frames_col, fps_col, steps_col = st.columns(3)
with frames_col:
    frames = st.select_slider("Frames", options=[9, 17, 33], value=17)
with fps_col:
    fps = st.select_slider("Frames per second", options=[4, 8, 16], value=8)
with steps_col:
    steps = st.select_slider("Inference steps", options=[4, 8, 16, 30], value=30)

if st.button(
    "Generate video",
    disabled="image_bytes" not in st.session_state,
    width="stretch",
):
    with st.status("Generating video", expanded=True) as status:
        status.write("Uploading the multipart request to Amazon S3")
        output_uri, failure_uri, request_key = submit_video(
            runtime,
            s3,
            state,
            video_prompt,
            st.session_state.image_bytes,
            num_frames=int(frames),
            fps=int(fps),
            steps=int(steps),
            seed=int(seed),
        )
        status.write("Waiting for the SageMaker asynchronous endpoint")
        started = time.perf_counter()
        try:
            st.session_state.video_bytes = wait_for_s3_object(
                s3,
                output_uri,
                failure_uri=failure_uri,
            )
        finally:
            s3.delete_object(Bucket=state.bucket, Key=request_key)
        validate_mp4(st.session_state.video_bytes)
        st.session_state.video_seconds = time.perf_counter() - started
        st.session_state.output_uri = output_uri
        status.update(label="Video ready", state="complete", expanded=False)

if "video_bytes" in st.session_state:
    st.video(st.session_state.video_bytes)
    st.caption(
        f"Wan VACE output in {st.session_state.video_seconds:.1f}s. "
        f"Stored at {st.session_state.output_uri}"
    )
    st.download_button(
        "Download video",
        st.session_state.video_bytes,
        "wan-vace.mp4",
        "video/mp4",
        width="stretch",
    )
