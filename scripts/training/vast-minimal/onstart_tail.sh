PYEOF
echo "[onstart] train.py escrito ($(wc -l < /workspace/train.py) lineas)"
echo "[onstart] instalando libs de sistema (libGL para opencv/ultralytics)"
apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 >/dev/null 2>&1 || echo "[onstart] WARN apt fallo (continuo)"
pip install -q --no-cache-dir "ultralytics>=8.4.46,<8.5" huggingface_hub onnx "onnxslim>=0.1.34" onnxruntime "numpy<2"
echo "[onstart] deps instaladas"
python -u /workspace/train.py
RC=$?
echo "[onstart] train.py exit=$RC"
echo "exit=$RC $(date -u)" > /workspace/DONE
echo "[onstart] FIN"
