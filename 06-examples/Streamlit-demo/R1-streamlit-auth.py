import streamlit as st
import yaml

class Authenticator:
    def __init__(self, config_path='config.yaml'):
        with open(config_path) as file:
            self.config = yaml.safe_load(file)

    def login(self):
        st.sidebar.title("Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        
        if st.sidebar.button("Login"):
            if username in self.config['credentials']:
                if password == self.config['credentials'][username]['password']:
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    # Clear messages on login
                    st.session_state['messages'] = []
                    st.rerun()
                else:
                    st.sidebar.error("Invalid password")
            else:
                st.sidebar.error("Invalid username")
        return False

    def logout(self):
        if st.sidebar.button("Logout"):
            # Clear all session state
            st.session_state['authenticated'] = False
            st.session_state['username'] = None
            st.session_state['messages'] = []
            st.rerun()
        return False