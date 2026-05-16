#!/usr/bin/env python3
"""Launcher local demo embebidos-3.

Verifica que la Nano responde, sirve el dashboard via http.server localhost:8001
(que getUserMedia acepta como contexto seguro), abre el browser y mantiene el
proceso vivo hasta Ctrl+C.

Uso:
  uv run --with requests python scripts/launch_demo.py
  uv run --with requests python scripts/launch_demo.py --nano-ip 100.100.166.120 --port 8001
"""
import argparse
import http.server
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path


def check_nano(nano_ip: str, port: int = 8000, timeout: float = 3.0) -> bool:
    """GET http://NANO_IP:8000/health. True si responde."""
    url = f"http://{nano_ip}:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            print(f"  health OK ({r.status}): {r.read()[:200].decode()}")
            return True
    except Exception as e:
        print(f"  health FAIL: {e}")
        return False


def serve_dashboard(dashboard_dir: Path, port: int) -> http.server.ThreadingHTTPServer:
    handler = http.server.SimpleHTTPRequestHandler

    class _H(handler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(dashboard_dir), **kw)

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), _H)
    t = threading.Thread(target=httpd.serve_forever, daemon=True, name="dashboard-http")
    t.start()
    return httpd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nano-ip", default="100.100.166.120", help="IP Tailscale o LAN de la Nano")
    p.add_argument("--nano-port", type=int, default=8000)
    p.add_argument("--port", type=int, default=8001, help="puerto local dashboard")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    dashboard_dir = Path(__file__).resolve().parent / "dashboard"
    if not (dashboard_dir / "index.html").exists():
        print(f"ERROR: no encontré {dashboard_dir / 'index.html'}", file=sys.stderr)
        sys.exit(1)

    print(f"[1/3] verificando Nano @ {args.nano_ip}:{args.nano_port}")
    nano_ok = check_nano(args.nano_ip, args.nano_port)
    if not nano_ok:
        print("  WARNING: la Nano no respondió. Arranco el dashboard igual; conectá manualmente cuando esté.")

    print(f"[2/3] sirviendo dashboard en http://localhost:{args.port}")
    try:
        httpd = serve_dashboard(dashboard_dir, args.port)
    except OSError as e:
        print(f"  ERROR: puerto {args.port} en uso ({e})")
        sys.exit(2)

    # inyectar la IP de la Nano via query string
    dashboard_url = f"http://localhost:{args.port}/?nano={args.nano_ip}:{args.nano_port}"
    if not args.no_browser:
        webbrowser.open(dashboard_url)

    print(f"[3/3] dashboard listo: {dashboard_url}")
    print(f"      WS target sugerido: ws://{args.nano_ip}:{args.nano_port}/ws")
    print(f"      Ctrl+C para salir")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nbye")
        httpd.shutdown()


if __name__ == "__main__":
    main()
