import json
from typing import Any, Dict


def to_json_report(result: Any) -> str:
    if hasattr(result, "to_dict"):
        return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    if isinstance(result, dict):
        return json.dumps(result, indent=2, ensure_ascii=False)
    return json.dumps({"result": str(result)}, indent=2, ensure_ascii=False)
