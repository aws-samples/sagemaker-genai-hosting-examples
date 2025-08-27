import logging
from typing import Any
from typing import Dict
from typing import Union

from constants import constants


logging.basicConfig(level=logging.INFO)


# Audio Parameters
AUDIO_INPUT = "audio_input"
LANGUAGE = "language"
TASK = "task"
FORCED_DECODER_IDS = "forced_decoder_ids"

# Text Generation parameters
MAX_LENGTH = "max_length"
NUM_RETURN_SEQUENCES = "num_return_sequences"
NUM_BEAMS = "num_beams"
TOP_P = "top_p"
EARLY_STOPPING = "early_stopping"
DO_SAMPLE = "do_sample"
NO_REPEAT_NGRAM_SIZE = "no_repeat_ngram_size"
TOP_K = "top_k"
TEMPERATURE = "temperature"
MIN_LENGTH = "min_length"
MIN_NEW_TOKENS = "min_new_tokens"
MAX_NEW_TOKENS = "max_new_tokens"
LENGTH_PENALTY = "length_penalty"
MAX_TIME = "max_time"


ALL_PARAM_NAMES = [
    AUDIO_INPUT,
    LANGUAGE,
    TASK,
    FORCED_DECODER_IDS,
    MAX_LENGTH,
    NUM_RETURN_SEQUENCES,
    NUM_BEAMS,
    TOP_P,
    EARLY_STOPPING,
    DO_SAMPLE,
    NO_REPEAT_NGRAM_SIZE,
    TOP_K,
    TEMPERATURE,
    MIN_LENGTH,
    MAX_NEW_TOKENS,
    MIN_NEW_TOKENS,
    LENGTH_PENALTY,
    MAX_TIME,
]

# Model parameter ranges
LENGTH_MIN = 1
NUM_RETURN_SEQUENCE_MIN = 1
NUM_BEAMS_MIN = 1
TOP_P_MIN = 0
TOP_P_MAX = 1
NO_REPEAT_NGRAM_SIZE_MIN = 1
TOP_K_MIN = 0
TEMPERATURE_MIN = 0
NEW_TOKENS_MIN = 0


def is_list_of_strings(parameter: Any) -> bool:
    """Return True if the parameter is a list of strings."""
    if parameter and isinstance(parameter, list):
        return all(isinstance(elem, str) for elem in parameter)
    else:
        return False


def _validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the parameters in the input loads.

    Checks if max_length, num_return_sequences, num_beams, top_p and temprature are in bounds.
    Checks if do_sample is boolean.
    Checks max_length, num_return_sequences and num_beams integers.

    Args:
        payload: a decoded input payload (dictionary of input parameter and values)

    Raises: ValueError is any of the check fails.
    """
    # For all parameters used in generation task, please see
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    for param_name in payload:
        if param_name not in ALL_PARAM_NAMES:
            raise ValueError(f"Input payload contains an invalid key '{param_name}'. Valid keys are {ALL_PARAM_NAMES}.")

    if AUDIO_INPUT not in payload:
        raise ValueError(f"Input payload must contain {AUDIO_INPUT} key.")

    if LANGUAGE in payload:
        value = payload[LANGUAGE]
        if type(value) != str:
            raise ValueError(f"{LANGUAGE} must be a string, got {value}.")
        value = value.lower()
        payload[LANGUAGE] = value
        if value not in constants.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Input payload contains an invalid language {value}. "
                f"Valid languages are {constants.SUPPORTED_LANGUAGES}."
            )
        if TASK not in payload:
            raise ValueError("Input payload should contain both language and task")

    if TASK in payload:
        value = payload[TASK]
        if type(value) != str:
            raise ValueError(f"{TASK} must be a string, got {value}.")
        value = value.lower()
        payload[TASK] = value
        if value not in constants.SUPPORTED_TASKS:
            raise ValueError(
                f"Input payload contains an invalid task {value}. Valid tasks are {constants.SUPPORTED_TASKS}."
            )
        if LANGUAGE not in payload:
            raise ValueError("Input payload should contain both language and task")

    for param_name in [MAX_LENGTH, NUM_RETURN_SEQUENCES, NUM_BEAMS]:
        if param_name in payload:
            if type(payload[param_name]) != int:
                raise ValueError(f"{param_name} must be an integer, got {payload[param_name]}.")

    if MAX_LENGTH in payload:
        if payload[MAX_LENGTH] < LENGTH_MIN:
            raise ValueError(f"{MAX_LENGTH} must be at least {LENGTH_MIN}, got {payload[MAX_LENGTH]}.")

    if MIN_LENGTH in payload:
        if payload[MIN_LENGTH] < LENGTH_MIN:
            raise ValueError(f"{MIN_LENGTH} must be at least {LENGTH_MIN}, got {payload[MIN_LENGTH]}.")

    if MAX_NEW_TOKENS in payload:
        if payload[MAX_NEW_TOKENS] < NEW_TOKENS_MIN:
            raise ValueError(f"{MAX_NEW_TOKENS} must be at least {NEW_TOKENS_MIN}, got {payload[MAX_NEW_TOKENS]}.")

    if MIN_NEW_TOKENS in payload:
        if payload[MIN_NEW_TOKENS] < NEW_TOKENS_MIN:
            raise ValueError(f"{MIN_NEW_TOKENS} must be at least {NEW_TOKENS_MIN}, got {payload[MIN_NEW_TOKENS]}.")

    if NUM_RETURN_SEQUENCES in payload:
        if payload[NUM_RETURN_SEQUENCES] < NUM_RETURN_SEQUENCE_MIN:
            raise ValueError(
                f"{NUM_RETURN_SEQUENCES} must be at least {NUM_RETURN_SEQUENCE_MIN}, "
                f"got {payload[NUM_RETURN_SEQUENCES]}."
            )

    if NUM_BEAMS in payload:
        if payload[NUM_BEAMS] < NUM_BEAMS_MIN:
            raise ValueError(f"{NUM_BEAMS} must be at least {NUM_BEAMS_MIN}, got {payload[NUM_BEAMS]}.")

    if NUM_RETURN_SEQUENCES in payload and NUM_BEAMS in payload:
        if payload[NUM_RETURN_SEQUENCES] > payload[NUM_BEAMS]:
            raise ValueError(
                f"{NUM_BEAMS} must be at least {NUM_RETURN_SEQUENCES}. Instead got "
                f"{NUM_BEAMS}={payload[NUM_BEAMS]} and {NUM_RETURN_SEQUENCES}="
                f"{payload[NUM_RETURN_SEQUENCES]}."
            )

    if TOP_P in payload:
        if payload[TOP_P] < TOP_P_MIN or payload[TOP_P] > TOP_P_MAX:
            raise ValueError(f"{TOP_K} must be in range [{TOP_P_MIN},{TOP_P_MAX}], got " f"{payload[TOP_P]}")

    if TEMPERATURE in payload:
        if payload[TEMPERATURE] < TEMPERATURE_MIN:
            raise ValueError(
                f"{TEMPERATURE} must be a float with value at least {TEMPERATURE_MIN}, got " f"{payload[TEMPERATURE]}."
            )

    if DO_SAMPLE in payload:
        if type(payload[DO_SAMPLE]) != bool:
            raise ValueError(f"{DO_SAMPLE} must be a boolean, got {payload[DO_SAMPLE]}.")

    return payload


def _update_num_beams(payload: Dict[str, Union[str, float, int]]) -> Dict[str, Union[str, float, int]]:
    """Add num_beans to the payload if missing and num_return_sequences is present.

    Args:
        payload (Dict): dictionary of input text and parameters
    Returns:
        payload (Dict): payload with number of beams updated
    """

    if NUM_RETURN_SEQUENCES in payload and NUM_BEAMS not in payload:
        payload[NUM_BEAMS] = payload[NUM_RETURN_SEQUENCES]
    return payload
