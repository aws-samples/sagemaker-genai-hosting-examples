# SGLang

SGLang is a high-performance serving framework for large language models and vision-language models. It is designed to deliver low-latency and high-throughput inference across a wide range of setups, from a single GPU to large distributed clusters. Its core features include:

- **Fast Backend Runtime**: Provides efficient serving with RadixAttention for prefix caching, a zero-overhead CPU scheduler, prefill-decode disaggregation, speculative decoding, continuous batching, paged attention, tensor/pipeline/expert/data parallelism, structured outputs, chunked prefill, quantization (FP4/FP8/INT4/AWQ/GPTQ), and multi-LoRA batching.

- **Extensive Model Support**: Supports a wide range of generative models (Llama, Qwen, DeepSeek, Kimi, GLM, GPT, Gemma, Mistral, etc.), embedding models (e5-mistral, gte, mcdse), and reward models (Skywork), with easy extensibility for integrating new models. Compatible with most Hugging Face models and OpenAI APIs.

- **Extensive Hardware Support**: Runs on NVIDIA GPUs (GB200/B300/H100/A100/Spark), AMD GPUs (MI355/MI300), Intel Xeon CPUs, Google TPUs, Ascend NPUs, and more.

- **Flexible Frontend Language**: Offers an intuitive interface for programming LLM applications, supporting chained generation calls, advanced prompting, control flow, multi-modal inputs, parallelism, and external interactions.

- **Active Community**: SGLang is open-source and supported by a vibrant community with widespread industry adoption, powering over 300,000 GPUs worldwide.


## Directory structure
- Docker build files (for EC2 and SageMaker Studio) and scripts to push container into ECR are in [docker](./docker)
- Inference examples using SGLang container for various models are in [examples](./examples)

## How to prepare SGLang container for SageMaker deployment
- If you are using EC2 instance please copy `Dockerfile`, `serve` and `build-ec2.sh` to the EC2 instance and run `build-ec2.sh`. These scripts were tested on `ml.g6e.4xlarge` instance with `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.4.1 (Ubuntu 22.04) 20240925` AMI.
- if you are using Amazon SageMaker AI Studio please clone the repo to the Studin environment and then run `studio_build.ipynb`


## Additional resources

- SGLang [documentation](https://docs.sglang.ai/)
- Official GitHub [repo](https://github.com/sgl-project/sglang)