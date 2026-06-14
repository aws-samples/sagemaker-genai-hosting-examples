import os
import json
import mlflow
from datetime import datetime
from typing import List, Dict
from langchain import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.llms import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.chains import LLMChain
from IPython.display import (
    display, 
    Markdown, 
    HTML
)


def pretty_print_html(text):
    # Replace newline characters with <br> tags
    html_text = text.replace('\n', '<br>')
    # Apply HTML formatting
    html_formatted = f'<pre style="font-family: monospace; background-color: #f8f8f8; padding: 10px; border-radius: 5px; border: 1px solid #0077b6;">{html_text}</pre>'
    # Display the formatted HTML
    return HTML(html_formatted)


def set_meta_llama_params(
    max_new_tokens=512,
    top_p=0.9,
    temperature=0.6,
):
    """ set Llama parameters """
    llama_params = {}
    llama_params['max_new_tokens'] = max_new_tokens
    llama_params['top_p'] = top_p
    llama_params['temperature'] = temperature
    return llama_params


def print_dialog(inputs, payload, response):
    dialog_output = []
    for msg in inputs:
        dialog_output.append(f"**{msg['role'].upper()}**: {msg['content']}\n")
    dialog_output.append(f"**ASSISTANT**: {response['generated_text']}")
    dialog_output.append("\n---\n")
    
    display(Markdown('\n'.join(dialog_output)))

def format_messages(messages: List[Dict[str, str]]) -> List[str]:
    """
    Format messages for Llama 3+ chat models.
    
    The model only supports 'system', 'user' and 'assistant' roles, starting with 'system', then 'user' and 
    alternating (u/a/u/a/u...). The last message must be from 'user'.
    """
    # auto assistant suffix
    # messages.append({"role": "assistant"})
    
    output = "<|begin_of_text|>"
    # Adding the inferred prefix
    _system_prefix = f"\n\nCutting Knowledge Date: December 2023\nToday Date: {datetime.now().strftime('%d %b %Y')}\n\n"
    for i, entry in enumerate(messages):
        output += f"<|start_header_id|>{entry['role']}<|end_header_id|>"
        if i == 0:
            output += f"{_system_prefix}{entry['content']}<|eot_id|>"
        elif i >= 1 and 'content' in entry:
            output += f"\n\n{entry['content']}<|eot_id|>"
    output += "<|start_header_id|>assistant<|end_header_id|>\n"
    return output


def write_eula(attribute):
    os.makedirs("/home/sagemaker-user/.license/", exist_ok=True)
    f = open("/home/sagemaker-user/.license/llama-license.txt", "w")
    f.write(attribute)
    f.close()
    return 0


def read_eula():
    attribute = open("/home/sagemaker-user/.license/llama-license.txt", "r").read()
    assert attribute == "True", f"Llama EULA set to {attribute}! Please review EULA to continue!"
    return attribute


class ContentHandlerwithTracking(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def __init__(self, experiment_name):
        self.mlflow_experiment_name = experiment_name
        print(f"Sending experiments to : {self.mlflow_experiment_name}")
        self.experiment_online_info = mlflow.set_experiment(self.mlflow_experiment_name)
        self.run_id_ephemeral = None

    def transform_input(self, prompt, model_kwargs):
        with mlflow.start_run(
            experiment_id=self.experiment_online_info.experiment_id,
            run_name=f"lc-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        ) as run:
            base_input = [{"role" : "user", "content" : prompt}]
            optz_input = format_messages(base_input)
            input_str = json.dumps(
                {
                    "inputs" : optz_input, 
                    "parameters" : {**model_kwargs}
                }
            )
            # track prompts
            # mlflow.log_param("SystemPrompt", instruction)
            mlflow.log_param("UserPrompt", optz_input)
            mlflow.log_param("parameters", {**model_kwargs})

            self.run_id_ephemeral = run.info.run_id

        return input_str.encode('utf-8')
    
    def transform_output(self, output):
        with mlflow.start_run(
            experiment_id=self.experiment_online_info.experiment_id,
            run_id=self.run_id_ephemeral
        ) as run:
            response_json = json.loads(output.read().decode("utf-8"))
            mlflow.log_param("ModelResponse", response_json["generated_text"])
        return response_json["generated_text"]
