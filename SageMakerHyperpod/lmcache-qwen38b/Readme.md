## Deploy Qwen3 model with Lmcache 


Apply the manifest file 

kubectl apply -f lmcache-qwen-8b.yaml 

# check the status of the pod and get pod ip
kubectl get pods -o wide 


# Once deployed, you can spin up a curl pod to run sample inference requests 

kubectl run curl-test \                                                                   
  --image=curlimages/curl:latest \
  --restart=Never \
  --rm -i --tty \
  -- /bin/sh



# Run sample requests(replace pod-ip with the ip adress of the pod) 
curl http://pod-ip:8000/v1/completions \
-H "Content-Type: application/json" \
-d "{
\"model\": \"deepseek-ai/DeepSeek-R1-0528-Qwen3-8B\",
\"prompt\": \"Making a chocolate and nut cake involves several steps, from preparing the batter to baking and decorating. Here's a simple yet delicious recipe for a chocolate and nut cake. This recipe assumes you're making a basic cake, but you can adjust it to your preference\",
\"max_tokens\": 100,
\"temperature\": 0
}"


# Check the logs of the pod 


kubectl logs -f podname  | grep -i "lmcache\|cache hit\|storing"

# Sample output that shows cache hit rate when using similar prompts

INFO 08-18 21:45:50 [loggers.py:122] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.1%, Prefix cache hit rate: 54.2%
[2025-08-18 21:49:46,163] LMCache INFO: Reqid: cmpl-1a7d41feffce420ba19525d93a33d165-0, Total tokens 51, LMCache hit tokens: 51, need to load: 2 (vllm_v1_adapter.py:803:lmcache.integration.vllm.vllm_v1_adapter)
[2025-08-18 21:49:46,164] LMCache INFO: Retrieved 51 out of 51 out of total 51 tokens (cache_engine.py:500:lmcache.v1.cache_engine)
[2025-08-18 21:49:49,672] LMCache WARNING: In connector.start_load_kv, but the attn_metadata is None (vllm_v1_adapter.py:446:lmcache.integration.vllm.vllm_v1_adapter)
INFO 08-18 21:49:50 [loggers.py:122] Engine 000: Avg prompt throughput: 5.1 tokens/s, Avg generation throughput: 10.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.1%, Prefix cache hit rate: 63.2%
INFO 08-18 21:50:00 [loggers.py:122] Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.1%, Prefix cache hit rate: 63.2%
