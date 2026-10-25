"""Test-wide settings isolation.

backend/.env may enable the live web / LLM fallback on a developer machine
(GENERAL_FALLBACK_MODE); tests must never make live calls, so it is forced
off here before any Settings object is created. Environment variables take
precedence over backend/.env in pydantic-settings. Tests that exercise the
fallback switch it on explicitly with monkeypatch.
"""

import os

os.environ["GENERAL_FALLBACK_MODE"] = "off"
os.environ["TAVILY_API_KEY"] = ""
os.environ["USAGE_TRACKING_PERSIST"] = "false"  # never write data/logs/usage.json
