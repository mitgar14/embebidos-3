#!/bin/bash
# Monitorea el onstart log buscando hitos / fin / errores.
# Uso: peek.sh <instance_id> [iteraciones=16] [intervalo_seg=30]
ID="${1:?uso: peek.sh <instance_id> [iter] [seg]}"
MAX="${2:-16}"
SEC="${3:-30}"
for i in $(seq 1 "$MAX"); do
  log=$(vastai logs "$ID" 2>/dev/null | tr -d '\0\r')
  last=$(printf '%s\n' "$log" | grep -vE '^\s*$' | tail -2 | tr '\n' '|')
  echo "[peek $i/$MAX] $last"
  if printf '%s\n' "$log" | grep -qE '\[DONE\]|train\.py exit='; then
    echo "=== FIN DETECTADO ==="
    break
  fi
  if printf '%s\n' "$log" | grep -qiE 'Traceback|Error:|CUDA error|Segmentation fault|No space left'; then
    echo "=== POSIBLE ERROR DETECTADO ==="
    break
  fi
  sleep "$SEC"
done
echo "=== tail 25 ==="
vastai logs "$ID" 2>/dev/null | tr -d '\0\r' | grep -vE '^\s*$' | tail -25
