"""
These functions were adapted from utils.py in LLMPerf
"""

import random
import math
from transformers import LlamaTokenizerFast
from typing import Any, Dict, List, Optional, Tuple

def sample_random_positive_int(mean: int, stddev: int) -> int:
    """Sample random numbers from a gaussian distribution until a positive number is sampled.

    Args:
        mean: The mean of the gaussian distribution to sample from.
        stddev: The standard deviation of the gaussian distribution to sample from.

    Returns:
        A random positive integer sampled from the gaussian distribution.
    """
    ret = -1
    while ret <= 0:
        ret = int(random.gauss(mean, stddev))
    return ret

def randomly_sample_sonnet_lines_prompt(
    openai_chat_completions: bool = True,
    prompt_tokens_mean: int = 800,
    prompt_tokens_stddev: int = 10,
    expect_output_tokens: int = 1000,
    tokenizer = LlamaTokenizerFast.from_pretrained(
        "hf-internal-testing/llama-tokenizer")
) -> Tuple[str, int]:
    """Generate a prompt that randomly samples lines from a the shakespeare sonnet at sonnet.txt.

    Args:
        prompt_length_mean: The mean length of the prompt to generate.
        prompt_len_stddev: The standard deviation of the length of the prompt to generate.
        expect_output_tokens: The number of tokens to expect in the output. This is used to
        determine the length of the prompt. The prompt will be generated such that the output
        will be approximately this many tokens.

    Note:
        tokens will be counted from the sonnet using the Llama tokenizer. Using one tokenizer
        ensures a fairer comparison across different LLMs. For example, if gpt 3.5 tokenizes
        a prompt in less tokens than Llama2, then this will be reflected in the results since
        they will be fed identical prompts.

    Returns:
        A tuple of the prompt and the length of the prompt.
    """

    get_token_length = lambda text: len(tokenizer.encode(text))

    prompt = (
        "Randomly stream lines from the following text "
        f"with {expect_output_tokens} output tokens. "
        "Don't generate eos tokens:\n\n"
    )
    # get a prompt length that is at least as long as the base
    num_prompt_tokens = sample_random_positive_int(
        prompt_tokens_mean, prompt_tokens_stddev
    )
    while num_prompt_tokens < get_token_length(prompt):
        num_prompt_tokens = sample_random_positive_int(
            prompt_tokens_mean, prompt_tokens_stddev
        )
    remaining_prompt_tokens = num_prompt_tokens - get_token_length(prompt)
    sonnet_path = "sonnet.txt"
    with open(sonnet_path, "r") as f:
        sonnet_lines = f.readlines()
    random.shuffle(sonnet_lines)
    sampling_lines = True
    while sampling_lines:
        for line in sonnet_lines:
            line_to_add = line
            if remaining_prompt_tokens - get_token_length(line_to_add) < 0:
                # This will cut off a line in the middle of a word, but that's ok since an
                # llm should be able to handle that.
                line_to_add = line_to_add[: int(math.ceil(remaining_prompt_tokens))]
                sampling_lines = False
                prompt += line_to_add
                break
            prompt += line_to_add
            remaining_prompt_tokens -= get_token_length(line_to_add)
    # Prepare a request with the selected schema
    if openai_chat_completions:
        request = {
              "messages": [
                {"role": "user", "content": prompt}
              ],
              "max_tokens": expect_output_tokens,
              "temperature": 0.75,
              "stop": None
            }
    else:
        request = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": expect_output_tokens,
                "temperature": 0.75
                }
            }

    return (request, num_prompt_tokens)

import time

def inference_latency(model,
                      openai_chat_completions: bool = True,
                      prompt_tokens_mean: int = 250,
                      prompt_tokens_stddev: int = 10,
                      expect_output_tokens: int = 500,
                      tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
                     ):
    error = False
    (request, _) = randomly_sample_sonnet_lines_prompt(
        openai_chat_completions,
        prompt_tokens_mean,
        prompt_tokens_stddev,
        expect_output_tokens,
        )
    start = time.time()
    try:
        results = model.predict(request)
    except:
        error = True
        results = []
    return {'latency': (time.time() - start)*1000.0, 'error': error, 'result': results}