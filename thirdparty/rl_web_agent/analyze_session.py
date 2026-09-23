#!/usr/bin/env python3
"""Inspect a recorded agent session.

Usage: python analyze_session.py <path/to/session.json>
"""
import json
import re
import sys

if len(sys.argv) != 2:
    sys.exit(f"usage: {sys.argv[0]} <session.json>")

# Load the session data
with open(sys.argv[1]) as f:
    data = json.load(f)

# Get an example of system message
print("SYSTEM MESSAGE:")
print(data["system_messages"][0]["text"][:200] + "...")
print("\n" + "-" * 80 + "\n")

# Get an example of assistant message with action
assistant_msgs = [msg for msg in data["conversation_history"] if msg["role"] == "assistant"]
print("ASSISTANT MESSAGE EXAMPLE:")
print(assistant_msgs[0]["content"][0]["text"])
print("\n" + "-" * 80 + "\n")

# Extract action from assistant message
action = None
for msg in assistant_msgs:
    content = msg["content"][0]["text"]
    action_match = re.search(r"ACTION: (.*?)(?:\n|$)", content)
    if action_match:
        action = action_match.group(1)
        break

print("EXTRACTED ACTION:")
print(action)
print("\n" + "-" * 80 + "\n")

# Parse action as JSON
action_json = json.loads(action)
print("ACTION AS JSON:")
print(json.dumps(action_json, indent=2))
