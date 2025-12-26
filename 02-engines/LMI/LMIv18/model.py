# adapters/my_adapter/model.py
from djl_python.input_parser import input_formatter

@input_formatter
def custom_input_formatter(decoded_payload: dict, tokenizer=None, **kwargs) -> dict:
    print(f"PAYLOAD: {decoded_payload}")
    if "messages" in decoded_payload:
        messages = decoded_payload["messages"]
        messages.append({'role': 'user', 'content': 'Speak like a pirate'})
        decoded_payload.update({"messages": messages})
    print(f"PAYLOAD: {decoded_payload}")
    
    return decoded_payload
