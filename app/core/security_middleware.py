from fastapi import Request, HTTPException, status
from typing import Callable
import logging
from app.core.config import settings
import re
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

class SecurityMiddleware:
    async def __call__(self, request: Request, call_next: Callable):
        try:
            # Rate limiting
            await self._check_rate_limit(request)
            
            # Input validation
            await self._validate_input(request)
            
            # Security headers
            response = await call_next(request)
            return self._add_security_headers(response)
            
        except Exception as e:
            logger.error(f"Security middleware error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
    
    async def _check_rate_limit(self, request: Request):
        """Implement rate limiting per IP"""
        client_ip = request.client.host
        current_time = datetime.now()
        
        # Simple in-memory rate limiting
        if not hasattr(self, '_rate_limit_store'):
            self._rate_limit_store = {}
            
        if client_ip in self._rate_limit_store:
            last_request_time = self._rate_limit_store[client_ip]
            time_diff = (current_time - last_request_time).total_seconds()
            
            if time_diff < settings.RATE_LIMIT_SECONDS:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many requests"
                )
                
        self._rate_limit_store[client_ip] = current_time
    
    async def _validate_input(self, request: Request):
        """Validate input data for security"""
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            
            # Validate content type
            if not content_type.startswith(("application/json", "multipart/form-data")):
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="Unsupported media type"
                )
            
            # Size validation for file uploads
            if content_type.startswith("multipart/form-data"):
                content_length = request.headers.get("content-length", 0)
                if int(content_length) > settings.MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="File too large"
                    )
    
    def _add_security_headers(self, response):
        """Add security headers to response"""
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }
        
        for header, value in headers.items():
            response.headers[header] = value
            
        return response 