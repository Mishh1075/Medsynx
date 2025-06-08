import logging
from datetime import datetime
from typing import Dict, Any, Optional
import json
from app.core.config import settings
import requests
from fastapi import Request
import hashlib

logger = logging.getLogger(__name__)

class AuditLogger:
    def __init__(self):
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        
        # Add file handler for audit logs
        handler = logging.FileHandler("audit.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
    async def log_event(
        self,
        event_type: str,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
        request: Optional[Request] = None
    ):
        """Log an audit event."""
        try:
            event = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                "user_id": user_id,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "action": action,
                "status": status,
                "details": details or {},
            }
            
            # Add request details if available
            if request:
                event["request_details"] = {
                    "method": request.method,
                    "url": str(request.url),
                    "client_ip": request.client.host,
                    "user_agent": request.headers.get("user-agent"),
                }
            
            # Log to file
            self.logger.info(json.dumps(event))
            
            # Send to monitoring webhook if configured
            if settings.MONITORING_WEBHOOK_URL and event_type in ["security", "privacy"]:
                await self._send_to_webhook(event)
                
            # Special handling for privacy-related events
            if event_type == "privacy":
                await self._handle_privacy_event(event)
                
        except Exception as e:
            logger.error(f"Error logging audit event: {str(e)}")
            
    async def _send_to_webhook(self, event: Dict[str, Any]):
        """Send event to monitoring webhook."""
        try:
            requests.post(
                settings.MONITORING_WEBHOOK_URL,
                json=event,
                timeout=5
            )
        except Exception as e:
            logger.error(f"Error sending to webhook: {str(e)}")
            
    async def _handle_privacy_event(self, event: Dict[str, Any]):
        """Special handling for privacy-related events."""
        try:
            # Check for privacy threshold violations
            if event.get("details", {}).get("privacy_score", 1.0) < settings.PRIVACY_METRICS_THRESHOLD:
                await self.log_event(
                    event_type="alert",
                    user_id=event["user_id"],
                    resource_type="privacy",
                    resource_id=event["resource_id"],
                    action="privacy_threshold_violation",
                    status="warning",
                    details={
                        "threshold": settings.PRIVACY_METRICS_THRESHOLD,
                        "actual_score": event["details"].get("privacy_score")
                    }
                )
        except Exception as e:
            logger.error(f"Error handling privacy event: {str(e)}")
            
    def get_audit_trail(
        self,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> list:
        """Retrieve audit trail with optional filters."""
        try:
            events = []
            with open("audit.log", "r") as f:
                for line in f:
                    event = json.loads(line.split(" - ")[-1])
                    
                    # Apply filters
                    if user_id and event["user_id"] != user_id:
                        continue
                    if resource_type and event["resource_type"] != resource_type:
                        continue
                    
                    event_date = datetime.fromisoformat(event["timestamp"])
                    if start_date and event_date < start_date:
                        continue
                    if end_date and event_date > end_date:
                        continue
                        
                    events.append(event)
                    
            return events
            
        except Exception as e:
            logger.error(f"Error retrieving audit trail: {str(e)}")
            return [] 