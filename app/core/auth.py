"""
Authentication Module

This module provides JWT and OAuth authentication for the application.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2AuthorizationCodeBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import requests
from app.core.config import settings
from app.db.models import User
from app.db.session import get_db
from sqlalchemy.orm import Session

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 configuration
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
oauth2_code_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl=settings.OAUTH_AUTH_URL,
    tokenUrl=settings.OAUTH_TOKEN_URL
)

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    refresh_token: Optional[str] = None

class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: list[str] = []

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)

def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create JWT access token.
    
    Args:
        data: Data to encode in token
        expires_delta: Token expiration time
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
        
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    return jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )

def create_refresh_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT refresh token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=30)
        
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })
    
    return jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    Get current user from JWT token.
    
    Args:
        token: JWT token
        db: Database session
        
    Returns:
        User object
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
        
    user = db.query(User).filter(User.email == token_data.username).first()
    if user is None:
        raise credentials_exception
        
    return user

async def get_oauth_token(
    code: str,
    provider: str
) -> Dict[str, Any]:
    """
    Get OAuth token from provider.
    
    Args:
        code: Authorization code
        provider: OAuth provider name
        
    Returns:
        Token response from provider
    """
    if provider == "google":
        return await _get_google_token(code)
    elif provider == "github":
        return await _get_github_token(code)
    else:
        raise ValueError(f"Unsupported OAuth provider: {provider}")

async def _get_google_token(code: str) -> Dict[str, Any]:
    """Get OAuth token from Google."""
    response = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": settings.GOOGLE_CLIENT_ID,
            "client_secret": settings.GOOGLE_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": settings.GOOGLE_REDIRECT_URI
        }
    )
    response.raise_for_status()
    return response.json()

async def _get_github_token(code: str) -> Dict[str, Any]:
    """Get OAuth token from GitHub."""
    response = requests.post(
        "https://github.com/login/oauth/access_token",
        headers={"Accept": "application/json"},
        data={
            "client_id": settings.GITHUB_CLIENT_ID,
            "client_secret": settings.GITHUB_CLIENT_SECRET,
            "code": code
        }
    )
    response.raise_for_status()
    return response.json()

async def get_oauth_user_info(
    token: str,
    provider: str
) -> Dict[str, Any]:
    """
    Get user info from OAuth provider.
    
    Args:
        token: OAuth access token
        provider: OAuth provider name
        
    Returns:
        User info from provider
    """
    if provider == "google":
        return await _get_google_user_info(token)
    elif provider == "github":
        return await _get_github_user_info(token)
    else:
        raise ValueError(f"Unsupported OAuth provider: {provider}")

async def _get_google_user_info(token: str) -> Dict[str, Any]:
    """Get user info from Google."""
    response = requests.get(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {token}"}
    )
    response.raise_for_status()
    return response.json()

async def _get_github_user_info(token: str) -> Dict[str, Any]:
    """Get user info from GitHub."""
    response = requests.get(
        "https://api.github.com/user",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }
    )
    response.raise_for_status()
    return response.json()

def require_scopes(required_scopes: list[str]):
    """
    Dependency for requiring specific scopes.
    
    Args:
        required_scopes: List of required scopes
        
    Returns:
        Dependency function
    """
    async def scope_validator(
        current_user: User = Depends(get_current_user)
    ) -> User:
        token_data = TokenData(
            username=current_user.email,
            scopes=current_user.scopes
        )
        
        for scope in required_scopes:
            if scope not in token_data.scopes:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required scope: {scope}"
                )
        return current_user
        
    return scope_validator 