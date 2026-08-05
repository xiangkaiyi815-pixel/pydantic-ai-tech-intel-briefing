from __future__ import annotations

import sys
import logging
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)

from search_assistant.mcp.public_sources import mcp


if __name__ == "__main__":
    mcp.run(transport="stdio")
