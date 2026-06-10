import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class UsageLogger:
    def __init__(self, path: str = "tokenopt_usage.jsonl"):
        self.path = Path(path)

    def log(self, payload: Dict[str, Any]) -> None:
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **payload}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
