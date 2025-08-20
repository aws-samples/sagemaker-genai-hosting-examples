## Sample file to deploy OpenAi-gptoss 20b model 

kubectl apply -f deploy-openai.yaml

# Verify Deployment 

kubectl get pods 

vggangas@842f5782fd40 ~ % kubectl get pods
NAME                                                 READY   STATUS    RESTARTS   AGE
gpt-oss-585cd7f9d7-w9b7h                             1/1     Running   0          7h21m



# sample curl request with the ALB url 


curl -X POST \
  'http://loadbalancer_dns:80/v1/completions' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openai/gpt-oss-20b",
    "prompt": "Once upon a time",
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": false
  }'            
{"id":"cmpl-ef7e2ce1eb894a3195bbaf0f037d5e3e","object":"text_completion","created":1755561583,"model":"openai/gpt-oss-20b","choices":[{"index":0,"text":", there was a little girl named Lily who lived in a small village. She was known for her bright smile and her love for storytelling. Lily would spend her afternoons with her family, telling stories about the world beyond the village. \n\nOne day, Lily decided to make a wish and she couldn't help but think about the people who had been suffering from a lack of resources like food and water. She was determined to find a way to bring help to them.”\n\n**Reason 2: The power of","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":4,"total_tokens":104,"completion_tokens":100,"prompt_tokens_details":null},"kv_transfer_params":null}%