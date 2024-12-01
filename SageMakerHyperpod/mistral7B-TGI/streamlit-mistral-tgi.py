import streamlit as st
import requests
import json
import time

# Streamlit app configuration
st.set_page_config(page_title="LLM Chatbot with Mistral 7B on SageMaker Hyperpod", page_icon="🤖")
st.title("🤖 LLM Chatbot with Mistral 7B on SageMaker Hyperpod")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response with streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Metrics containers
        metrics_container = st.empty()
        start_time = time.time()
        first_token_received = False
        token_count = 0

        try:
            # Send the request to the TGI container via the Load Balancer
            # Replace LB_DNS with teh DNS name of NLB that was created. You can find that by going to Aws Console->Ec2->Load Balancers
            url = "http://LB_DNS:80/generate"
            headers = {"Content-Type": "application/json"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 256,
                    "temperature": 0.5
                }
            }
            response = requests.post(url, headers=headers, json=payload, stream=True)

            # Stream the response
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        if "generated_text" in chunk:
                            token_text = chunk["generated_text"]
                            token_count += len(token_text.split())
                            if not first_token_received:
                                ttft = time.time() - start_time
                                first_token_received = True

                            full_response += token_text
                            message_placeholder.markdown(full_response + "▌")
                    except json.JSONDecodeError:
                        continue

            # Final update without cursor
            message_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"Error: {str(e)}")
            full_response = "Sorry, there was an error generating the response."

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Add a clear button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Display some information about the app
with st.sidebar:
    st.markdown("""
    ### About
    This is a chatbot interface for the Mistral model deployed on SageMaker HyperPod.
    
    ### Model Parameters
    - Temperature: 0.5
    - Max New Tokens: 256
    """)
