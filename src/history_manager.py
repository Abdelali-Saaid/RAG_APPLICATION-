import json
import os
import glob
from datetime import datetime
from .config import BASE_DIR

HISTORY_DIR = os.path.join(BASE_DIR, "history")
os.makedirs(HISTORY_DIR, exist_ok=True)

class HistoryManager:
    """Manages persistent chat history sessions."""
    
    @staticmethod
    def save_session(session_id: str, messages: list):
        """Save a list of messages to a JSON file."""
        if not messages:
            return
            
        file_path = os.path.join(HISTORY_DIR, f"{session_id}.json")
        data = {
            "session_id": session_id,
            "updated_at": datetime.now().isoformat(),
            "messages": messages
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_session(session_id: str):
        """Load messages from a JSON file."""
        file_path = os.path.join(HISTORY_DIR, f"{session_id}.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("messages", [])
        return []

    @staticmethod
    def list_sessions():
        """Return a list of all session metadata, sorted by date."""
        files = glob.glob(os.path.join(HISTORY_DIR, "*.json"))
        sessions = []
        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    sessions.append({
                        "id": data["session_id"],
                        "updated_at": data["updated_at"],
                        "preview": data["messages"][0]["content"][:30] + "..." if data["messages"] else "Empty Chat"
                    })
            except Exception:
                continue
        
        # Sort by updated_at descending
        return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)

    @staticmethod
    def delete_session(session_id: str):
        """Delete a session file."""
        file_path = os.path.join(HISTORY_DIR, f"{session_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
