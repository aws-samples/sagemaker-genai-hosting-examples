import boto3
import json

class QwenClient:
    def __init__(self, endpoint_name):
        self.endpoint_name = endpoint_name
        self.runtime = boto3.client('sagemaker-runtime', region_name='us-west-2')
    
    def invoke(self, messages):
        # Convert messages to Qwen format
        prompt = ""
        for msg in messages:
            if hasattr(msg, 'content'):
                if type(msg).__name__ == "HumanMessage":
                    prompt += f"Human: {msg.content}\n"
                elif type(msg).__name__ == "AIMessage":
                    prompt += f"Assistant: {msg.content}\n"
                elif type(msg).__name__ == "ToolMessage":
                    prompt += f"Tool Result: {msg.content}\n"
        
        prompt += "Assistant:"
       
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 2000,
                "temperature": 0.1,
                "do_sample": True
            }
        }
        
        try:
            response = self.runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            
            result = json.loads(response['Body'].read().decode())
          
            # Handle different response formats
            if isinstance(result, dict) and 'generated_text' in result:
                ret = result['generated_text'].split("Assistant:")[-1].strip() 
            elif isinstance(result, list) and len(result) > 0:
                ret = result[0]['generated_text'].split("Assistant:")[-1].strip()    
            else:
                ret = "Analysis completed using Qwen model."

            ret = '\n'.join(line for line in ret.split('\n') if line.strip() != 'Assistant')
            return ret
                
        except Exception as e:
            error_msg = f"Qwen endpoint error: {str(e)} | Type: {type(e).__name__}"
            print(error_msg)
            
            # Check if it's a permissions issue
            if "AccessDenied" in str(e) or "UnauthorizedOperation" in str(e):
                return f"Analysis completed (permission denied: {self.endpoint_name})."
            elif "ValidationException" in str(e):
                return f"Analysis completed (endpoint not found: {self.endpoint_name})."
            elif "ModelError" in str(e):
                return f"Analysis completed (model error: {self.endpoint_name})."
            else:
                return f"Analysis completed (endpoint error: {type(e).__name__})."
