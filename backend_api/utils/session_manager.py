"""
Session Management System
Handles user sessions, data persistence, and cleanup
"""

import uuid
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import shutil

class SessionManager:
    """Manages user sessions with data persistence"""
    
    def __init__(self, session_timeout_hours=24, data_dir="./session_data"):
        self.session_timeout = timedelta(hours=session_timeout_hours)
        self.data_dir = Path(data_dir)
        self.sessions = {}
        
        # Create session data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing sessions from disk
        self._load_sessions()
    
    def create_session(self):
        """Create a new session and return session_id"""
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        
        session_data = {
            "session_id": session_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_accessed": datetime.utcnow().isoformat(),
            "data": {}
        }
        
        self.sessions[session_id] = session_data
        self._save_session(session_id)
        
        print(f"✅ Created session: {session_id}")
        return session_id
    
    def get_session(self, session_id):
        """Get session data, return None if not found"""
        if session_id not in self.sessions:
            return None
        
        # Update last accessed time
        self.sessions[session_id]["last_accessed"] = datetime.utcnow().isoformat()
        self._save_session(session_id)
        
        return self.sessions[session_id]
    
    def save_data(self, session_id, key, value):
        """Save data to session"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        self.sessions[session_id]["data"][key] = value
        self._save_session(session_id)
    
    def get_data(self, session_id, key, default=None):
        """Get data from session"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        return self.sessions[session_id]["data"].get(key, default)
    
    def delete_session(self, session_id):
        """Delete a session"""
        if session_id not in self.sessions:
            return False
        
        del self.sessions[session_id]
        
        # Delete session data file
        session_file = self.data_dir / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
        
        print(f"🗑️  Deleted session: {session_id}")
        return True
    
    def _save_session(self, session_id):
        """Save session to disk"""
        session_file = self.data_dir / f"{session_id}.json"
        
        try:
            # Don't save actual data objects, just metadata
            session_metadata = {
                "session_id": self.sessions[session_id]["session_id"],
                "created_at": self.sessions[session_id]["created_at"],
                "last_accessed": self.sessions[session_id]["last_accessed"],
                "data_keys": list(self.sessions[session_id]["data"].keys())
            }
            
            with open(session_file, 'w') as f:
                json.dump(session_metadata, f, indent=2)
        except Exception as e:
            print(f"⚠️  Error saving session {session_id}: {str(e)}")
    
    def _load_sessions(self):
        """Load sessions from disk"""
        if not self.data_dir.exists():
            return
        
        for session_file in self.data_dir.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                    session_id = session_data.get("session_id")
                    
                    # Reinitialize session in memory
                    self.sessions[session_id] = {
                        "session_id": session_id,
                        "created_at": session_data.get("created_at"),
                        "last_accessed": session_data.get("last_accessed"),
                        "data": {}
                    }
            except Exception as e:
                print(f"⚠️  Error loading session from {session_file}: {str(e)}")
    
    def cleanup_expired_sessions(self):
        """Remove sessions that have expired"""
        current_time = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session_data in self.sessions.items():
            last_accessed = datetime.fromisoformat(session_data["last_accessed"])
            
            if current_time - last_accessed > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.delete_session(session_id)
            print(f"⏰ Expired and removed session: {session_id}")
        
        return len(expired_sessions)
    
    def get_session_count(self):
        """Get number of active sessions"""
        return len(self.sessions)
