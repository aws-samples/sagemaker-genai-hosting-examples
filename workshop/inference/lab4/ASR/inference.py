import ast
import io
import json
import logging
from copy import deepcopy
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import audio_validation
import numpy as np
import torch
from audio_validation import _update_num_beams
from audio_validation import _validate_payload
from constants import constants
from sagemaker_inference import encoder
from scipy.io.wavfile import read
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from transformers.pipelines.audio_utils import ffmpeg_read


SAMPLE_RATE = 16000


class ModelAndProcessor:
    """An ASR model with explicit model and tokenizer objects."""

    def __init__(self, model_dir: str) -> None:
        """Initialize model with provided model kwargs and processor objects."""
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        self.model = self.model.to(device)
        self.model.config.forced_decoder_ids = None
        self.model.eval()
        logging.info("Loaded model")

        self.processor = WhisperProcessor.from_pretrained(model_dir)
        logging.info("Loaded processor")

    def __call__(self, audio_input: Dict, **kwargs: Any) -> List:
        """Perform inference via calls to processor and model's generate method.

        If the model is loaded on the GPU, input_ids are placed on the GPU device context.
        """
        input_features = self.processor(
            audio_input["raw"], sampling_rate=audio_input["sampling_rate"], return_tensors="pt"
        ).input_features

        if next(self.model.parameters()).is_cuda:
            input_ids_device = input_features.cuda()
        else:
            input_ids_device = input_features

        if kwargs:
            if audio_validation.LANGUAGE in kwargs:
                language = kwargs.pop(audio_validation.LANGUAGE)
                task = kwargs.pop(audio_validation.TASK)
                kwargs[audio_validation.FORCED_DECODER_IDS] = self.processor.get_decoder_prompt_ids(
                    language=language, task=task
                )

        predicted_ids = self.model.generate(input_ids_device, **kwargs)

        outputs = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return {constants.TEXT: outputs}


def model_fn(model_dir: str) -> Tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """Create our inference task as a delegate to the model.

    This runs only once per one worker.

    Args:
        model_dir (str): directory where the model files are stored.
    Returns:
        WhisperForConditionalGeneration: a huggingface model for Automatic Speech Recognition.
        WhisperProcessor: a huggingface processor for pre-process the audio inputs and post-process the model outputs.

    Raises:
        ValueError if the model file cannot be found.
    """
    try:
        return ModelAndProcessor(model_dir)
    except Exception:
        logging.exception(f"Failed to load model from: {model_dir}")
        raise


def transform_fn(
    audio_generator_processor: ModelAndProcessor,
    input_data: bytes,
    content_type: str,
    accept: str,
) -> bytes:
    """Make predictions against the model and return a serialized response.

    The function signature conforms to the SM contract.

    Args:
        audio_generator_processor: a huggingface pipeline
        input_data (obj): the request data.
        content_type (str): the request content type.
        accept (str): accept header expected by the client.
    Returns:
        obj: a byte string of the prediction.
    """

    if content_type == constants.AUDIO_WAV:
        try:
            data = ffmpeg_read(input_data, SAMPLE_RATE)
            audio_input = {"sampling_rate": SAMPLE_RATE, "raw": data}
        except Exception:
            logging.exception(
                f"Failed to parse input payload. For content_type= {constants.AUDIO_WAV}, input "
                f"payload must be a bytearray"
            )
            raise
        try:
            output = audio_generator_processor(deepcopy(audio_input))
        except Exception:
            logging.exception("Failed to do inference")
            raise

    elif content_type == constants.APPLICATION_JSON:
        try:
            payload = json.loads(input_data)
        except Exception:
            logging.exception(
                f"Failed to parse input payload. For content_type={constants.APPLICATION_JSON}, input "
                f"payload must be a json encoded dictionary with keys {audio_validation.ALL_PARAM_NAMES}."
            )
            raise
        payload = _validate_payload(payload)
        payload = _update_num_beams(payload)
        audio_input = payload.pop(audio_validation.AUDIO_INPUT)
        audio_input = ffmpeg_read(bytes.fromhex(audio_input), SAMPLE_RATE)

        audio_input = {"sampling_rate": SAMPLE_RATE, "raw": audio_input}

        try:
            output = audio_generator_processor(deepcopy(audio_input), **payload)
        except Exception:
            logging.exception("Failed to do inference")
            raise
    else:
        raise ValueError('{{"error": "unsupported content type {}"}}'.format(content_type or "unknown"))
    if accept.endswith(constants.VERBOSE_EXTENSION):
        accept = accept.rstrip(constants.VERBOSE_EXTENSION)  # Verbose and non-verbose response are identical
    return encoder.encode(output, accept)
