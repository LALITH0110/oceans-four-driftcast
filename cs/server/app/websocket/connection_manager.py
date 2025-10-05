"""
WebSocket connection management for real-time client communication
"""
from fastapi import WebSocket
from typing import Dict, List, Any
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.client_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept WebSocket connection and register client"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.client_metadata[client_id] = {
            "connected_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "message_count": 0
        }
        logger.info(f"Client {client_id} connected via WebSocket")
    
    def disconnect(self, client_id: str):
        """Remove client connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.client_metadata:
            del self.client_metadata[client_id]
        logger.info(f"Client {client_id} disconnected from WebSocket")
    
    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                await websocket.send_json(message)
                
                # Update metadata
                if client_id in self.client_metadata:
                    self.client_metadata[client_id]["last_activity"] = datetime.utcnow()
                    self.client_metadata[client_id]["message_count"] += 1
                
                logger.debug(f"Message sent to client {client_id}: {message.get('type', 'unknown')}")
                
            except Exception as e:
                logger.error(f"Error sending message to client {client_id}: {e}")
                # Remove broken connection
                self.disconnect(client_id)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
                
                # Update metadata
                if client_id in self.client_metadata:
                    self.client_metadata[client_id]["last_activity"] = datetime.utcnow()
                    self.client_metadata[client_id]["message_count"] += 1
                    
            except Exception as e:
                logger.error(f"Error broadcasting to client {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up broken connections
        for client_id in disconnected_clients:
            self.disconnect(client_id)
        
        logger.info(f"Broadcast message sent to {len(self.active_connections)} clients")
    
    async def send_to_multiple(self, message: Dict[str, Any], client_ids: List[str]):
        """Send message to multiple specific clients"""
        for client_id in client_ids:
            await self.send_personal_message(message, client_id)
    
    def get_connected_clients(self) -> List[str]:
        """Get list of connected client IDs"""
        return list(self.active_connections.keys())
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    def get_client_metadata(self, client_id: str) -> Dict[str, Any]:
        """Get metadata for specific client"""
        return self.client_metadata.get(client_id, {})
    
    def is_client_connected(self, client_id: str) -> bool:
        """Check if client is connected"""
        return client_id in self.active_connections
    
    async def ping_all_clients(self):
        """Send ping to all clients to check connectivity"""
        ping_message = {
            "type": "ping",
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.broadcast(ping_message)
    
    async def send_system_announcement(self, announcement: str, priority: str = "info"):
        """Send system announcement to all clients"""
        message = {
            "type": "system_announcement",
            "message": announcement,
            "priority": priority,
            "timestamp": datetime.utcnow().isoformat()
        }
        await self.broadcast(message)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        now = datetime.utcnow()
        total_messages = sum(
            metadata.get("message_count", 0) 
            for metadata in self.client_metadata.values()
        )
        
        return {
            "total_connections": len(self.active_connections),
            "total_messages": total_messages,
            "clients": {
                client_id: {
                    "connected_duration": (now - metadata["connected_at"]).total_seconds(),
                    "last_activity": metadata["last_activity"].isoformat(),
                    "message_count": metadata["message_count"]
                }
                for client_id, metadata in self.client_metadata.items()
            }
        }

# Global connection manager instance
connection_manager = ConnectionManager()
