#!/usr/bin/env python3
"""Lee el JSON de `vastai search offers --raw` por stdin y muestra lo esencial."""
import json
import sys

data = json.load(sys.stdin)
offers = data["offers"] if isinstance(data, dict) and "offers" in data else data

print(f"{'ID':>9}  {'GPU':16}  {'$/h':>7}  {'rel':>5}  {'cuda':>5}  "
      f"{'inet_dn':>7}  {'dlperf':>7}  geo")
for o in offers[:12]:
    print(f"{o.get('id',''):>9}  {str(o.get('gpu_name','')):16}  "
          f"{o.get('dph_total',0):>7.3f}  {o.get('reliability2',0):>5.3f}  "
          f"{str(o.get('cuda_max_good','')):>5}  {o.get('inet_down',0):>7.0f}  "
          f"{o.get('dlperf',0):>7.1f}  {o.get('geolocation','')}")
