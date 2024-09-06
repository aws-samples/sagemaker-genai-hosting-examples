import os
import subprocess
import requests
from pathlib import Path
import time

working_dir = "/home/model-server"
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(working_dir,"weights")
model_name = "llava-v1.5-7b"
# url for the weights mirror
REPLICATE_WEIGHTS_URL = "https://weights.replicate.delivery/default"
HUGGING_FACE_WEIGHTS_URL = "https://huggingface.co/liuhaotian"
# files to download from the weights mirrors
weights = [
    {
        "dest": os.path.join(working_dir,model_name),
        # git commit hash from huggingface
        "src": "llava-v1.5-7b/resolve/main",
        "base": HUGGING_FACE_WEIGHTS_URL,
        "files": [
            "config.json",
            "generation_config.json",
            "pytorch_model-00001-of-00002.bin",
            "pytorch_model-00002-of-00002.bin",
            "pytorch_model.bin.index.json",
            "special_tokens_map.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ]
    },
    {
        "dest": os.path.join(working_dir,"openai/clip-vit-large-patch14-336"),
        "src": "clip-vit-large-patch14-336/ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
        "base": REPLICATE_WEIGHTS_URL,
        "files": [
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin"
        ],
    }
]


def _download_json(url: str, dest: Path):
    res = requests.get(url, allow_redirects=True)
    if res.status_code == 200 and res.content:
        with dest.open("wb") as f:
            f.write(res.content)
    else:
        print(f"Failed to download {url}. Status code: {res.status_code}")


def _download_weights(baseurl: str, basedest: str, host: str, files: list[str]):
    basedest = Path(basedest)
    start = time.time()
    print(f"downloading to: {basedest} and host {host} ")
    basedest.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = basedest / f
        url = os.path.join(host, baseurl, f)
        if not dest.exists():
            print("downloading url: ", url)
            if dest.suffix == ".json":
                _download_json(url, dest)
            else:
                subprocess.check_call(["pget", url, str(dest)], close_fds=False)
    print("downloading took: ", time.time() - start)


if __name__ == '__main__':
    for w in weights:
        print(f"base is {w['base']}")
        _download_weights(w["src"], w["dest"], w["base"], w["files"])