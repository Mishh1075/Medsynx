import streamlit as st
import requests
from typing import Optional, Dict, Any

class AuthManager:
    def __init__(self, api_url: str):
        self.api_url = api_url
        
    def login(self, email: str, password: str) -> bool:
        """
        Authenticate user and store token in session state.
        """
        try:
            response = requests.post(
                f"{self.api_url}/api/v1/auth/token",
                data={"username": email, "password": password}
            )
            
            if response.status_code == 200:
                token_data = response.json()
                st.session_state["token"] = token_data["access_token"]
                st.session_state["is_authenticated"] = True
                return True
            else:
                return False
        except Exception as e:
            st.error(f"Login error: {str(e)}")
            return False
            
    def register(self, email: str, password: str) -> bool:
        """
        Register a new user.
        """
        try:
            response = requests.post(
                f"{self.api_url}/api/v1/auth/register",
                json={"email": email, "password": password}
            )
            
            if response.status_code == 200:
                return True
            else:
                st.error(response.json()["detail"])
                return False
        except Exception as e:
            st.error(f"Registration error: {str(e)}")
            return False
            
    def logout(self):
        """
        Clear authentication state.
        """
        if "token" in st.session_state:
            del st.session_state["token"]
        if "is_authenticated" in st.session_state:
            del st.session_state["is_authenticated"]
            
    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """
        Get current user information.
        """
        if not self.is_authenticated():
            return None
            
        try:
            response = requests.get(
                f"{self.api_url}/api/v1/auth/me",
                headers=self.get_auth_headers()
            )
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
            
    def is_authenticated(self) -> bool:
        """
        Check if user is authenticated.
        """
        return st.session_state.get("is_authenticated", False)
        
    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for API requests.
        """
        if not self.is_authenticated():
            return {}
            
        return {
            "Authorization": f"Bearer {st.session_state['token']}"
        }
        
def render_auth_ui(auth_manager: AuthManager):
    """
    Render authentication UI components.
    """
    if not auth_manager.is_authenticated():
        st.sidebar.title("Authentication")
        tab1, tab2 = st.sidebar.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")
                
                if submit:
                    if auth_manager.login(email, password):
                        st.success("Logged in successfully!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
                        
        with tab2:
            with st.form("register_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                submit = st.form_submit_button("Register")
                
                if submit:
                    if password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        if auth_manager.register(email, password):
                            st.success("Registration successful! Please login.")
                            
    else:
        user = auth_manager.get_current_user()
        if user:
            st.sidebar.text(f"Logged in as: {user['email']}")
            if st.sidebar.button("Logout"):
                auth_manager.logout()
                st.rerun() 