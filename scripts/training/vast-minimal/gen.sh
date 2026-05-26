#!/bin/bash
# Ensambla onstart.sh = head + train.py (embebido en heredoc) + tail, con finales LF.
set -e
cat onstart_head.sh train.py onstart_tail.sh > onstart.sh
sed -i 's/\r$//' onstart.sh
echo "onstart.sh OK: $(wc -l < onstart.sh) lineas; marcadores PYEOF: $(grep -c '^PYEOF$' onstart.sh)"
