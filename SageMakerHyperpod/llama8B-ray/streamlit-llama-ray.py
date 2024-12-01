import streamlit as st
import requests
import json
import time

# Streamlit app configuration
st.set_page_config(page_title="Llama 3 8B Chat on Ray Serve", page_icon="🦙")
st.title("🦙 Llama 3 8B Chat Assistant")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# Display chat history
for message in st.session_state.messages:
    if message["role"] != "system":  # Don't display system messages
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
        
        # Metrics containers
        metrics_container = st.empty()
        start_time = time.time()

        try:
            # Send the request to the Ray Serve endpoint
            # Replace LB_DNS with teh DNS name of NLB that was created. You can find that by going to Aws Console->Ec2->Load Balancers
            url = "http://LB_DNS/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": "meta-llama/Meta-Llama-3-8B-Instruct",
                "messages": st.session_state.messages,  # Send full conversation history
                "temperature": 0.7
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                response_data = response.json()
                assistant_response = response_data['choices'][0]['message']['content']
                
                # Display the response
                message_placeholder.markdown(assistant_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_response
                })
                
                # Display metrics
                end_time = time.time()
                with metrics_container:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Response Time", f"{(end_time - start_time):.2f}s")
                    with col2:
                        st.metric("Response Length", len(assistant_response.split()))

            else:
                st.error(f"Error: Received status code {response.status_code}")
                st.error(response.text)

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Add a clear button
if st.button("Clear Chat"):
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.rerun()

# Display information about the app
with st.sidebar:
    st.markdown("""
    ### About
    This is a chat interface for the Llama 3 8B model deployed on Ray Serve with vLLM.
    
    ### Model Details
    - Model: Meta-Llama-3-8B-Instruct
    - Temperature: 0.7
    - Backend: Ray Serve with vLLM
    
    ### Features
    - Full conversation history
    - Response metrics
    - Chat-like interface
    """)

# Optional: Add a download button for chat history
if st.sidebar.button("Download Chat History"):
    chat_history = "\n\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in st.session_state.messages
        if msg['role'] != 'system'
    ])
    st.sidebar.download_button(
        label="Download",
        data=chat_history,
        file_name="chat_history.txt",
        mime="text/plain"
    )