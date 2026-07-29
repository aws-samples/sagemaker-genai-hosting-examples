# Kimi K3 with SageMaker HyperPod Inference Operator

To deploy Kimi K3 on HyperPod using the HyperPod Inference operator, you need an ml.p6-b300.48xlarge instance.

Deploy using 
```
kubectl apply -f kimi-k3.yaml
```

This will download the model weights from Hugging Face and run on vLLM.

To invoke, get the DNS name for the created `Ingress` object using `kubectl get ingress` and invoke using the OpenAI SDK such as:

```python
import time
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="<ALB DNS>/v1",
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Introduce Kimi K3 in one sentence"
            }
        ]
    }
]

start = time.time()
response = client.chat.completions.create(
    model="Kimi-K3",
    messages=messages,
    max_tokens=2048
)
```
