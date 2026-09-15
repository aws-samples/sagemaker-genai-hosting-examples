#!/usr/bin/env python3
"""Generate an image with FLUX.2-klein and animate it with Wan VACE."""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path

import boto3

from vllm_omni_media import (
    DEFAULT_STATE_PATH,
    invoke_image,
    load_state,
    submit_video,
    validate_mp4,
    wait_for_s3_object,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-prompt",
        default=(
            "Cinematic photograph of a small observatory on a windswept coastal "
            "cliff at sunrise, detailed clouds, natural light"
        ),
    )
    parser.add_argument(
        "--video-prompt",
        default=(
            "Slow camera push-in toward the coastal observatory as clouds drift "
            "across the sky and ocean waves move below; preserve the building, "
            "coastline, and composition"
        ),
    )
    parser.add_argument("--image-size", default="1024x1024")
    parser.add_argument("--image-steps", type=int, default=4)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--video-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_STATE_PATH,
    )
    parser.add_argument(
        "--keep-request",
        action="store_true",
        help="Keep the multipart request object in S3 after generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = load_state(args.state_file)
    session = boto3.Session(region_name=state.region)
    runtime = session.client("sagemaker-runtime")
    s3 = session.client("s3")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")

    image_started = time.perf_counter()
    image_bytes = invoke_image(
        runtime,
        state,
        args.image_prompt,
        size=args.image_size,
        steps=args.image_steps,
        seed=args.seed,
    )
    image_seconds = time.perf_counter() - image_started
    image_path = args.output_dir / f"{stamp}-flux2-klein.png"
    image_path.write_bytes(image_bytes)
    print(f"Image: {image_path} ({image_seconds:.1f}s)")

    video_started = time.perf_counter()
    output_uri, failure_uri, request_key = submit_video(
        runtime,
        s3,
        state,
        args.video_prompt,
        image_bytes,
        width=args.width,
        height=args.height,
        num_frames=args.frames,
        fps=args.fps,
        steps=args.video_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    try:
        video_bytes = wait_for_s3_object(
            s3,
            output_uri,
            failure_uri=failure_uri,
            timeout_seconds=args.timeout,
        )
    finally:
        if not args.keep_request:
            s3.delete_object(Bucket=state.bucket, Key=request_key)

    validate_mp4(video_bytes)
    video_seconds = time.perf_counter() - video_started
    video_path = args.output_dir / f"{stamp}-wan-vace.mp4"
    video_path.write_bytes(video_bytes)
    print(f"Video: {video_path} ({video_seconds:.1f}s)")
    print(f"S3 output: {output_uri}")


if __name__ == "__main__":
    main()
