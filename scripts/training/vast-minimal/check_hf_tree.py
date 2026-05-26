#!/usr/bin/env python3
"""Lista los archivos del repo HF (JSON del API tree por stdin) ordenados por path."""
import sys
import json

data = json.load(sys.stdin)
files = [x for x in data if x.get("type") == "file"]
for x in sorted(files, key=lambda e: e.get("path", "")):
    size = x.get("size", 0)
    print(f"{size:>10}  {x.get('path')}")
print(f"--- {len(files)} archivos ---")
