import platform
import sys
import psutil
import json
from pathlib import Path

output = Path("results/system")
output.mkdir(parents=True, exist_ok=True)

info = {
    "python_version": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "processor": platform.processor(),
    "logical_cpu_count": psutil.cpu_count(logical=True),
    "physical_cpu_count": psutil.cpu_count(logical=False),
    "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
}

with open(output / "system_info.json", "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

print(json.dumps(info, indent=2, ensure_ascii=False))