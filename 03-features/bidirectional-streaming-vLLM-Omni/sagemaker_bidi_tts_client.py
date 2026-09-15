"""Gradio client for streaming vLLM-Omni TTS on SageMaker AI."""

import argparse

import gradio as gr
import numpy as np
from sagemaker_bidi_tts import stream_pcm_events


async def stream_audio(
    endpoint_name,
    region,
    text,
    voice,
    language,
):
    """Yield Gradio audio chunks and progress text."""
    if not text or not text.strip():
        raise gr.Error("Enter text to synthesize.")

    total_bytes = 0
    chunk_count = 0
    async for event in stream_pcm_events(
        endpoint_name=endpoint_name,
        text=text.strip(),
        region=region,
        voice=voice.strip() or "Vivian",
        language=language,
    ):
        if event.get("type") != "audio.chunk":
            continue

        pcm = np.frombuffer(event["audio"], dtype=np.int16).copy()
        total_bytes += len(event["audio"])
        chunk_count += 1
        status = f"Streaming chunk {chunk_count}, {total_bytes:,} audio bytes received."
        yield (event["sample_rate"], pcm), status

    if chunk_count == 0:
        raise gr.Error("The endpoint returned no audio.")


def build_demo(endpoint_name, region):
    """Build the interactive text-to-speech application."""

    async def synthesize(text, voice, language):
        async for output in stream_audio(
            endpoint_name,
            region,
            text,
            voice,
            language,
        ):
            yield output

    with gr.Blocks(title="Streaming TTS with vLLM-Omni") as demo:
        gr.Markdown("# Streaming text-to-speech")
        gr.Markdown(
            "Generate speech with vLLM-Omni through an Amazon SageMaker AI "
            "bidirectional stream."
        )
        text = gr.Textbox(
            label="Text",
            value=(
                "Hello, this response is streaming from vLLM-Omni "
                "on Amazon SageMaker AI."
            ),
            lines=4,
            max_lines=8,
        )
        with gr.Row():
            voice = gr.Textbox(label="Voice", value="Vivian")
            language = gr.Dropdown(
                label="Language",
                choices=["English", "Chinese", "Auto"],
                value="English",
            )
        generate = gr.Button("Generate speech", variant="primary")
        audio = gr.Audio(
            label="Streamed audio",
            streaming=True,
            autoplay=True,
        )
        status = gr.Textbox(label="Status", interactive=False)

        generate.click(
            synthesize,
            inputs=[text, voice, language],
            outputs=[audio, status],
        )

    return demo


def main():
    """Parse command-line options and launch Gradio."""
    parser = argparse.ArgumentParser(
        description="Stream vLLM-Omni TTS from a SageMaker endpoint."
    )
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--server-port", type=int, default=6006)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_demo(args.endpoint_name, args.region)
    demo.queue().launch(
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
