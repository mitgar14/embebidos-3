# scripts/labeling — Plan B: auto-labeling en RTX 3060 de Nicolás

Servicio FastAPI que corre **Autodistill + Grounding DINO** en WSL2 sobre la RTX 3060 de Nicolás, expuesto al tailnet vía Tailscale Serve. Replica el patrón de `scripts/server/` del Jetson Nano (job persistence, SSE logs, sudoers granular) adaptado a una máquina con GPU.

**Cuándo usar este Plan B**: solo si Vast.ai cae, los créditos se agotan, o queremos infra permanente post-demo. Por defecto, el flujo primario es Vast.ai on-demand — ver `investigaciones/2026-05-20-infra-auto-label-remoto.md`.

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│ Windows 11 host (Nicolas)                                       │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │ WSL2 Ubuntu 22.04                                         │  │
│   │                                                            │  │
│   │  systemd: embebidos3-label.service                        │  │
│   │    └─ uvicorn → scripts.labeling.server.main:app          │  │
│   │         (FastAPI :8765, SSE logs, job persistence)        │  │
│   │                                                            │  │
│   │  /opt/embebidos3-label/   ← codigo                        │  │
│   │  /var/lib/embebidos3-label/jobs/  ← artifacts persistentes│  │
│   │  /run/embebidos3-label/  ← state volatil (jobs.json)      │  │
│   │  /etc/embebidos3-label.env  ← HF_TOKEN (chmod 600 root)   │  │
│   │                                                            │  │
│   │  conda env: embebidos3-label (python 3.10 + torch + cuda) │  │
│   │  GPU: /usr/lib/wsl/lib/nvidia-smi (driver via host)       │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Tailscale (Windows host, NO en WSL2 - rompe MTU)               │
│    └─ tailscale serve --bg 8765 (desde WSL2)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Tailnet (cifrado WireGuard)
                              ▼
┌──────────────────────────┐   ┌──────────────────────────┐
│ mitgar14 (Windows)       │   │ Claude Code (agente)     │
│  https://nicolas-pc.     │   │  requests / SSE          │
│    tailnet-X.ts.net/     │   │  POST /autolabel/job     │
│    health                │   │  GET  /jobs/.../logs     │
└──────────────────────────┘   └──────────────────────────┘
```

## Pasos manuales para Nicolás (one-shot, ~45 min)

### Pre-requisitos en el host Windows

1. **Tailscale en Windows** — descargar de [tailscale.com/download](https://tailscale.com/download), login.
   - Pedir a mitgar14 que comparta su tailnet (Invite User) o usar cuenta común.
   - **NO instalar Tailscale dentro de WSL2** — rompe MTU (issues #4140, #4833).
2. **WSL2 con systemd activo**. En WSL2 Ubuntu (no PowerShell):
   ```bash
   sudo tee /etc/wsl.conf > /dev/null <<EOL
   [boot]
   systemd=true
   EOL
   ```
   Luego desde PowerShell:
   ```powershell
   wsl --shutdown
   wsl  # reabrir
   ```
3. **Driver NVIDIA del host Windows actualizado** (≥570 recomendado). Verificar en WSL2:
   ```bash
   /usr/lib/wsl/lib/nvidia-smi
   ```
   Debe mostrar tu RTX 3060 sin "CUDA Version: ERR!".

### Instalación (idempotente, re-ejecutable sin daño)

1. Clonar el repo en tu home de WSL2:
   ```bash
   git clone <REPO_URL> ~/embebidos-3
   cd ~/embebidos-3
   ```
2. Lanzar el bootstrap:
   ```bash
   bash scripts/labeling/install/install_wsl2.sh
   ```
   Te pedirá el `HF_TOKEN` (input oculto). Pegar el que te pase mitgar14 — se guarda en `/etc/embebidos3-label.env` con permisos `600 root:root`.

3. El script:
   - Verifica systemd + CUDA.
   - Instala Miniconda si no está.
   - Crea conda env `embebidos3-label` (puede tardar 5-10 min — descarga PyTorch + CUDA + autodistill).
   - Copia el repo a `/opt/embebidos3-label/` (tu working copy queda intacta).
   - Instala servicio systemd + lo arranca.
   - Expone vía Tailscale Serve si está instalado.

### Verificación

```bash
# Estado del servicio
sudo systemctl status embebidos3-label

# Logs en vivo
sudo journalctl -u embebidos3-label -f

# Smoke test
curl http://localhost:8765/health
curl http://localhost:8765/system   # devuelve GPU info + activeJobs

# URL accesible desde el tailnet (mitgar14, Claude)
tailscale status     # buscar tu DNSName
```

### Mantenimiento

- **Reiniciar servicio**: `sudo systemctl restart embebidos3-label`
- **Pararlo temporalmente** (libera GPU para juegos):
  ```bash
  sudo systemctl stop embebidos3-label
  # Cuando termines:
  sudo systemctl start embebidos3-label
  ```
- **Cancelar job activo desde la propia PC**:
  ```bash
  curl -X DELETE http://localhost:8765/jobs/<JOB_ID>
  ```
- **Logs de un job específico**:
  ```bash
  cat /var/lib/embebidos3-label/jobs/<JOB_ID>/worker.log
  ```

### Tu privacidad

- El servicio escucha sólo en `127.0.0.1:8765` por defecto. Tailscale Serve hace el bridge al tailnet — sólo dispositivos autorizados de tu tailnet acceden.
- Puedes **parar el servicio** en cualquier momento (`sudo systemctl stop embebidos3-label`).
- No hay escritorio remoto, no hay shell remoto. Solo los endpoints HTTP definidos.
- Logs de jobs viven en `/var/lib/embebidos3-label/jobs/` (los puedes borrar cuando quieras).
- HF token está en `/etc/embebidos3-label.env` con `chmod 600 root:root` — sólo root lo lee.

## API (consumida por mitgar14 + Claude desde el tailnet)

| Endpoint | Función |
|---|---|
| `GET /health` | Ping |
| `GET /system` | nvidia-smi (gpu, vram, util, temp, activeJobs) |
| `POST /autolabel/job` | Dispara job. Body: `{input_url, ontology, conf?, model_type?, hf_dataset_repo?, hf_dataset_path?}` |
| `GET /jobs` | Lista jobs (activos + históricos) |
| `GET /jobs/{id}/state` | Estado: running, done, failed, cancelled |
| `GET /jobs/{id}/logs` | SSE stream de logs en vivo |
| `GET /jobs/{id}/artifact` | ZIP con labels YOLO (cuando state=done) |
| `DELETE /jobs/{id}` | Cancela job (SIGTERM al worker) |

Concurrency: **1 job activo a la vez** (RTX 3060 6 GB no aguanta 2 GroundingDINO simultáneos). POST returna 409 si hay otro corriendo.

## Cliente desde mitgar14 / Claude

```python
import requests

BASE = "https://nicolas-pc.<TAILNET>.ts.net"   # o http://localhost:8765 si pruebas local
r = requests.post(f"{BASE}/autolabel/job", json={
    "input_url": "hf://mitgar14/embebidos3-raw-batches/batch1",
    "ontology": {
        "plastic bottle or plastic container": "plastic",
        "paper or cardboard": "paper",
        "glass bottle or glass jar": "glass",
    },
    "conf": 0.25,
    "model_type": "tiny",       # cabe en 6 GB
    "hf_dataset_repo": "mitgar14/embebidos3-labels",
    "hf_dataset_path": "batch1",
})
job_id = r.json()["job_id"]

# Poll state
while True:
    state = requests.get(f"{BASE}/jobs/{job_id}/state").json()["state"]
    print(state)
    if state in ("done", "failed", "cancelled"):
        break
    time.sleep(5)

# Bajar artifact ZIP local
with requests.get(f"{BASE}/jobs/{job_id}/artifact", stream=True) as r:
    with open("labels.zip", "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
```

## Troubleshooting

| Síntoma | Causa probable | Fix |
|---|---|---|
| `systemd no esta activo` | `/etc/wsl.conf` sin `systemd=true` | Activarlo y `wsl --shutdown` |
| `nvidia-smi: command not found` | PATH sin `/usr/lib/wsl/lib` | El service unit ya lo incluye en `Environment=PATH=...`. Verificar que se uso la unit correcta. |
| `CUDA Version: ERR!` en nvidia-smi | Driver Windows desactualizado | Actualizar driver NVIDIA del host (>=570) |
| OOM al cargar GroundingDINO | RTX 3060 6 GB + procesos GPU en Windows | `nvidia-smi` para ver qué consume VRAM, cerrar OBS/juegos. Usar `model_type=tiny` (no base). |
| 409 al POST /autolabel/job | Job activo en curso | Esperar o `DELETE /jobs/{id}` |
| Tailscale serve no funciona | Doble Tailscale (Windows host + WSL2) | Desinstalar Tailscale de WSL2; dejar sólo el del host. |

## Referencias

- Investigación: `investigaciones/2026-05-20-infra-auto-label-remoto.md` (Ronda 1).
- Patrón fuente: `scripts/server/` (dashboard embebidos3 del Jetson Nano).
- Autodistill: <https://github.com/autodistill/autodistill-grounding-dino>.
- Tailscale Serve: <https://tailscale.com/kb/1242/tailscale-serve>.
- WSL2 CUDA: <https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl>.
