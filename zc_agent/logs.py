import json
import os
import secrets
from pathlib import Path
from datetime import datetime

from pydantic_ai.messages import ModelMessagesTypeAdapter

def serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


class ConversationJsonLogger:
    def __init__(self, log_dir):
        log_dir = log_dir
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

    @staticmethod
    def log_entry(agent, messages, source="user", extra=None):
        tools = []
        
        for ts in agent.toolsets:
            tools.extend(ts.tools.keys())

        dict_messages = ModelMessagesTypeAdapter.dump_python(messages)

        return {
            "agent_name": agent.name,
            "system_prompt": agent._instructions,
            "provider": agent.model.system,
            "model": agent.model.model_name,
            "tools": tools,
            "messages": dict_messages,
            "source": source,
            "extra": extra or {},
        }

    def log(self, agent, messages, source='user', extra=None) -> Path:
        entry = self.log_entry(agent, messages, source, extra)

        ts = entry['messages'][-1]['timestamp']
        ts_str = ts.strftime("%Y%m%d_%H%M%S")

        rand_hex = secrets.token_hex(3)
        filename = f"{agent.name}_{ts_str}_{rand_hex}.json"
        
        filepath = self.log_dir / filename

        with filepath.open("w", encoding="utf-8") as f_out:
            json.dump(entry, f_out, indent=2, default=serializer)

        return filepath

    def list_logs(self):
        log_files = sorted(self.log_dir.glob("*.json"))
        return log_files
