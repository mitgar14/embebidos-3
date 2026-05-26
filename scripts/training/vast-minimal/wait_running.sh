#!/bin/bash
# Espera a que la instancia este running, luego muestra el tail del onstart log.
# Uso: wait_running.sh <instance_id>
ID="${1:?uso: wait_running.sh <instance_id>}"
for i in $(seq 1 18); do
  st=$(vastai show instance "$ID" --raw 2>/dev/null | tr -d '\0\r' | grep -m1 '"actual_status"')
  echo "[wait $i] $st"
  if echo "$st" | grep -q 'running'; then
    echo "[wait] instancia running."
    break
  fi
  sleep 20
done
echo "=== logs (tail 40) ==="
vastai logs "$ID" 2>&1 | tr -d '\0\r' | tail -40
