import streamlit as st
import boto3
import json
import time
from auth import Authenticator
from logger import Logger

import os

# Set AWS region
os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'

# Initialize SageMaker runtime client with the specified region
client = boto3.client('sagemaker-runtime', region_name='us-west-2')

# Must be the first Streamlit command
st.set_page_config(page_title="Chatbot with DeepSeek R1", page_icon="🤖")

# Initialize authentication and logging
auth = Authenticator()
logger = Logger(cloudwatch_log_group='/chatbot/interactions')

# Initialize messages in session state if not exists
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Authentication check
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    auth.login()
    st.stop()

# Show logout button
auth.logout()

# Initialize SageMaker runtime client
client = boto3.client('sagemaker-runtime')

# Display title
st.title("🤖 DeepSeek R1 Chatbot")
st.markdown(f"Logged in as: {st.session_state['username']}")
st.markdown("---")

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

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Metrics containers
        metrics_container = st.empty()
        start_time = time.time()
        #add SM endpoint below
        try:
            response = client.invoke_endpoint_with_response_stream(
                EndpointName="",
                ContentType="application/json",
                Body=json.dumps({
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4000,
                    "temperature": 0.6,
                    "top_p": 0.9,
                    "stream": True
                })
            )

            for event in response['Body']:
                chunk = json.loads(event['PayloadPart']['Bytes'].decode())
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    content = chunk['choices'][0].get('delta', {}).get('content', '')
                    full_response += content
                    message_placeholder.markdown(full_response + "▌")

            # Final update without cursor
            message_placeholder.markdown(full_response)

        except Exception as e:
            error_message = f"Error: {str(e)}"
            st.error(error_message)
            logger.logger.error(f"Error generating response: {str(e)}")
            full_response = "Sorry, there was an error generating the response."

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Calculate and display metrics
        end_time = time.time()
        total_time = end_time - start_time
        metrics_container.markdown(f"Response time: {total_time:.2f} seconds")

        # Log the interaction
        logger.log_interaction(
            username=st.session_state['username'],
            prompt=prompt,
            response=full_response,
            response_time=total_time
        )

# Add a clear button
if st.button("Clear Chat"):
    st.session_state.messages = []
    logger.logger.info(f"Chat cleared by user: {st.session_state['username']}")
    st.rerun()

# Display some information about the app
with st.sidebar:
    st.markdown("""
    ### About
    This is a chatbot interface for the DeepSeek R1 model deployed on SageMaker.
    
    ### Model Parameters
    - Temperature: 0.6
    - Top P: 0.9
    - Streaming: Enabled
    
    ### DeepSeek R1
    R1 is a LLM developed by DeepSeek. It's known for its impressive reasoning capabilities.
    """)

    # Add usage statistics
    st.markdown("### Session Statistics")
    message_count = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.markdown(f"Messages sent: {message_count}")

    # Add version information
    st.markdown("### App Information")
    st.markdown("Version: 1.0.0")
    st.markdown(f"Current User: {st.session_state['username']}")

# Add footer
st.markdown("---")