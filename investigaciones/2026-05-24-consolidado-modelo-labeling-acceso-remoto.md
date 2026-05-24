# Consolidado técnico — Modelo, auto-labeling remoto y acceso remoto UAO

**Proyecto:** `embebidos-3` — clasificador YOLOv8n FP16 de 3 clases (`paper`/`glass`/`plastic`) sobre banda transportadora, deployado en Jetson Nano. Demo **2026-05-26**.
**Consolida 4 investigaciones** (2026-05-15 modelo · 2026-05-20 auto-labeling · 2026-05-22 Tailscale UAO · 2026-05-23 Headscale). Se mantiene lo esencial: decisiones, accionables, hallazgos empíricos y estado de infraestructura. Las tablas exhaustivas de fuentes, catálogos de YouTube e hipótesis descartadas viven en los documentos originales.

**Hardware/runtime de referencia:** Jetson Nano B01, JetPack 4.6.1 (L4T R32.7.1, Ubuntu 18.04 ARM64), Python 3.6.9 system, TensorRT 8.2.1.8, Maxwell `sm_53`. Engine `best_fp16.engine` 13 MB, 43 FPS / 23 ms a `imgsz=416` (shape fijo). Cámara Logitech C920.

---

## Parte I — Mejoras al modelo de detección (recall en small objects)

### Problema observado (2026-05-15)

Escena de testing en lab UAO: 3 piezas pequeñas de plástico sobre fondo de tela cremosa. **Solo 1 detección** (`plástico 51%`), 2 falsos negativos (paquete amarillo parcial + sobre verde claro de bajo contraste).

**Diagnóstico multi-causal** (NO es el engine TRT):
- (a) `conf` default 0,25 demasiado alto.
- (b) Padding mismatch: Roboflow exporta con **Fit-black=0**, Ultralytics usa **LetterBox=114** por defecto (sin medir hasta ahora).
- (c) `imgsz=416` insuficiente para objetos pequeños.
- (d) Escasa representación del dominio real (banda + tela cremosa + iluminación lab) en el dataset Roboflow `waste-3class-lwld8` v1-B (17.910 train / 1.739 valid / 844 test).

### Quick wins priorizados por costo × impacto

| # | Acción | Tiempo | Δ recall esperado | Recompila engine | Riesgo |
|---|---|---|---|---|---|
| 1 | `conf=0.10`, `iou=0.45` en `model.predict()` del server | 2 min | **+15-25 pts** | No | FP bajo en escena limpia |
| 2 | Verificar padding 0 vs 114 (script abajo) | 15 min | +5-15 si hay mismatch | No | Ninguno |
| 3 | SAHI con `supervision` (tiles 208×208, overlap 0,4) | 30 min | **+20-35 pts** small | No | Latencia ×3-4 → 10-14 FPS |
| 4 | TTA offline en `best.pt` (medir techo de recall) | 30 min | medición | No (solo `.pt`) | TTA no corre en TRT |
| 5 | Capturar ~150 imgs del setup real C920 + label Roboflow | 4-6 h | **+10-30 pts** dominio | No | Tiempo de labeling |
| 6 | Fine-tune 50 épocas RTX 4090 (`freeze=10`) | 2-3 h | **+10-30 pts** | Sí (ONNX nuevo) | Overfit si <100 imgs |
| 7 | Re-export ONNX + recompilar engine `imgsz=640` | 1-1,5 h | +5-15 pts | Sí | FPS 43 → 25-30 |
| 8 | Re-validar end-to-end, **recall@conf=0.10 por clase** | 2 h | confirmación | — | Métrica prioritaria, no mAP global |

**Métrica prioritaria:** `Recall@conf=0.10` por clase (no `mAP@0.5:0.95`). En aplicación industrial, no detectar es el error crítico. Curva PR canónica: `model.val(conf=0.001, iou=0.6, plots=True)` → buscar punto donde `R ≥ 0.90` por clase.

### Snippets esenciales

**1. Inferencia con threshold bajado (server FastAPI):**
```python
from ultralytics import YOLO
model = YOLO("best.engine")
results = model.predict(frame_np, conf=0.10, iou=0.45,
                        imgsz=416, half=True, device=0, verbose=False)
```

**2. SAHI tiled inference sobre el engine TRT** (la técnica más potente sin re-train):
```python
import supervision as sv
from ultralytics import YOLO
model = YOLO("best.engine")

def slicer_callback(slice_img):
    r = model.predict(slice_img, conf=0.10, iou=0.45,
                      imgsz=416, half=True, device=0, verbose=False)[0]
    return sv.Detections.from_ultralytics(r)

slicer = sv.InferenceSlicer(
    callback=slicer_callback, slice_wh=(208, 208),
    overlap_ratio_wh=(0.4, 0.4),
    overlap_filter_strategy=sv.OverlapFilter.NON_MAX_MERGE,
    iou_threshold=0.3)
detections = slicer(frame)
```
> **Gotcha Nano:** `supervision` requiere Python ≥3.7; el system Python del Nano es 3.6.9. Verificar el venv del server. Si solo hay 3.6, implementar el slicer manual con NumPy (split del frame en tiles con overlap → predict por tile → trasladar bboxes al frame → NMS global con `cv2.dnn.NMSBoxes`). Latencia estimada en Nano: ~9-14 FPS efectivos (cumple threshold MVP de 10 FPS).

**3. Verificación del padding mismatch** (sospecha #1, 15 min):
```python
import cv2
from ultralytics import YOLO

def letterbox(img, new_shape, color):
    h, w = img.shape[:2]
    r = min(new_shape[0]/h, new_shape[1]/w)
    nw, nh = int(w*r), int(h*r)
    resized = cv2.resize(img, (nw, nh))
    pw, ph = new_shape[1]-nw, new_shape[0]-nh
    t, b = ph//2, ph-ph//2; l, rr = pw//2, pw-pw//2
    return cv2.copyMakeBorder(resized, t, b, l, rr, cv2.BORDER_CONSTANT, value=color)

img = cv2.imread("test_3plasticos.jpg")
model = YOLO("best.engine")
for pad in [(0,0,0), (114,114,114)]:
    r = model.predict(letterbox(img, (416,416), pad), conf=0.001, iou=0.45, verbose=False)[0]
    print(f"pad={pad[0]}: {len(r.boxes)} dets, conf={r.boxes.conf.mean():.3f}")
```
Si `pad=0` da mejor resultado → el server debe usar `color=(0,0,0)` (matching training). Si `pad=114` gana → re-exportar Roboflow v2 con preprocessing "Letterbox".

**4. Fine-tune corto desde `best.pt`** (hiperparámetros clave):
```python
model = YOLO("best.pt").train(
    data="waste-3class-v2.yaml", imgsz=640, epochs=50, batch=32,
    lr0=0.0005, lrf=0.01, optimizer="AdamW",
    mosaic=1.0, copy_paste=0.5, mixup=0.15, hsv_v=0.6, hsv_s=0.5,
    scale=0.7, fliplr=0.5, flipud=0.0, degrees=5.0,
    close_mosaic=10, freeze=10, patience=15, device=0)
```
- `freeze=10`: congela el backbone (capas 0-9), solo actualiza neck + head → **anti-catastrophic forgetting** (crítico para no olvidar las 17.910 imágenes iniciales).
- `copy_paste=0.5` + `hsv_v=0.6`: clave para plásticos translúcidos sobre fondos cremosos nuevos con iluminación variable.
- `close_mosaic=10`: estabiliza la convergencia final con imágenes realistas.

### Decisión: NO migrar de versión de YOLO

| Modelo | mAP@0.5 small (DOTAv1.5) | TRT 8.2 export | Veredicto |
|---|---|---|---|
| **YOLOv8n** | **67,88%** | Limpio, opset 13 | **Mantener (baseline actual)** |
| YOLOv9t | 61,71% | Workarounds menores | Sin ganancia clara |
| YOLOv10n | 51,16% | **BLOQUEADO** — operador `fmod`/Mod no soportado en TRT 8.2.1.8 | **Descartar** |
| YOLOv11n | 64,33% | Parcial, requiere opset≥15 | Riesgo alto, ganancia incierta |

Fuente: Tariq & Javed 2025 ([arXiv:2504.09900](https://arxiv.org/abs/2504.09900)). El módulo C2f de v8 preserva resolución espacial mejor que la atención de v11 cuando hay muchos objetos pequeños por imagen.

### Negative constraints (modelo)

- **NO** aplicar augmentation doble (Roboflow al exportar + Ultralytics en train) → distribución contaminada. Roboflow solo preprocessing; Ultralytics todo el augmentation.
- **NO** usar `dynamic=True` en export TRT sobre Maxwell JP 4.6.1 (TRT 8.x inestable con shapes dinámicos). Compilar engines fijos separados (416 streaming, 640 validación).
- **NO** migrar a YOLOv10 (operador `fmod` no soportado en TRT 8.2; requiere TRT ≥8.4 o cirugía del grafo ONNX).
- **NO** modificar la arquitectura YAML (P2 head, attention) en el plazo del demo — aunque SOD-YOLOv8 demuestra +4,5 pts mAP@0.5 ([arXiv:2408.04786](https://arxiv.org/abs/2408.04786)), el re-train + recompilación no se justifica. Registrar como ablation propuesta en el informe IEEE.
- **TTA** no funciona en engines TRT (solo `.pt`). Útil offline para diagnóstico, no para producción.
- YOLOv8 **no expone `class_weights`** en API pública → sobre-muestrear la clase deficitaria en Roboflow.
- Hard negative mining **no es nativo** → flujo manual con FiftyOne + duplicar FN en `train/`.

### Aportes para el informe IEEE

1. Diagnóstico cuantitativo del **padding mismatch** Roboflow vs Ultralytics (único registro conocido para waste + Roboflow + Nano).
2. Comparativa empírica `conf=0.25` vs `conf=0.10` por clase sobre escena real.
3. **SAHI sobre TRT FP16 en Maxwell `sm_53`** — combinación no documentada en literatura (la mayoría usa Volta+ o `.pt`). Registro empírico de FPS/recall sería novedoso.
4. Fine-tune con `freeze=10` + augmentation reforzado — ablation por componente.

---

## Parte II — Infraestructura de auto-labeling remoto

**Tarea:** orquestar jobs de auto-labeling (Autodistill + Grounding DINO) sin GPU local. mitgar14 (Windows sin GPU) + Nicolás (RTX 3060 6 GB, no siempre presente) + Claude Code (agente).

### Veredicto

- **PRIMARIO: Vast.ai on-demand** con notebook `autolabel_vastai.ipynb` análogo al de training. Costo despreciable (RTX 3060 12 GB ≈ $0,06/hr, job de ~20-30 min ≈ $0,02-0,03), autonomía total para Claude (SDK + REST), cero fricción de coordinación, no toca la PC de Nicolás.
- **PLAN B (standby): FastAPI + systemd-in-WSL2 + Tailscale Serve** en la RTX 3060 de Nicolás. Activar solo si se agotan créditos Vast.ai o se quiere infraestructura permanente post-demo. La RTX 3060 6 GB queda muy justa (Grounding DINO base ~4-5 GB + procesos Windows).

### Snippet Vast.ai SDK (provisión autónoma + auto-destroy)

```python
from vastai import VastAI
vast = VastAI()  # lee ~/.config/vastai/vast_api_key

offers = vast.search_offers(
    query="gpu_name=RTX_3060 gpu_ram>=10 reliability>0.95 rentable=true",
    order="dph_total+", limit="3")

vast.create_instance(
    id=offers[0]["id"],
    image="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
    disk=20, ssh=True, direct=True,
    onstart_cmd=(
        "pip install -q autodistill autodistill-grounding-dino "
        "supervision huggingface_hub && "
        "python /workspace/autolabel.py && "
        "vastai destroy instance $CONTAINER_API_KEY"))
```
`CONTAINER_API_KEY` lo inyecta Vast.ai con scope limitado (start/stop/destroy solo de esa instancia) — no expone la API key principal. Para jobs efímeros: **destroy, no stop** (el storage sigue facturando si solo se detiene).

### Resultado de la ejecución end-to-end (smoke test 2026-05-20)

**100% recall (37/37 imágenes), 256 bboxes, ~$0,02 total.** Instancia RTX 3060 12 GB (Polonia), runtype `ssh_direct`.

Distribución de bboxes generados: **paper 56% (143) · plastic 25% (65) · glass 19% (48)**. Esto **invierte** la dominancia del dataset v1-B (donde el modelo overfitteó plastic 5,3×) → es exactamente lo que necesita el modelo para des-sesgar. Valida la intuición de re-equilibrar el dataset.

**Bugs corregidos en el notebook** (API frágil de `autodistill-grounding-dino`):
- `GroundingDINO(...)` ya **no acepta `model_type`** ni `conf` en `.label()`: pasar `box_threshold` + `text_threshold` al constructor (heurística `text = box × 0.7`).
- El **gate de calidad debe iterar `train/labels` + `valid/labels`** (Autodistill hace auto-split; contar solo `train` reporta 78% cuando el recall real es 100%).
- Pin de deps transitivas: `transformers<5` (transformers 5+ rompe `AutoTokenizer`), `scikit-learn`, `roboflow`, `timm>=1.0`, `accelerate`, `opencv-python-headless`.
- Migrar `huggingface-cli` → `hf download`/`hf upload` con `--token` explícito (no se hereda en subshell SSH).
- Instancias `running`: usar SSH directo, no `vastai execute`. `vastai destroy` interactivo: `echo "y" | vastai destroy instance ID`.

**Artifacts publicados:** HF datasets privados `mitgar14/embebidos3-raw-batches` (input) y `mitgar14/embebidos3-labels` (output: `data.yaml` + train/valid pairs).

### Negative constraints (labeling)

- **NO** usar `best.pt` como pre-labeler (refuerza el overfit hacia plastic).
- **NO** doble augmentation Roboflow + Ultralytics (mismo constraint que Parte I).
- **NO** doble Tailscale (Windows host + WSL2) → rompe MTU. Tailscale solo en el host; exponer puerto WSL2 con `tailscale serve --bg PORT`.
- **NO** instalar drivers NVIDIA dentro de WSL2 (usa el del host vía `/usr/lib/wsl/lib/`). SSH no-interactivo no carga PATH → usar ruta absoluta o `bash -lc`.

---

## Parte III — Acceso remoto a la Nano en UAO (Tailscale → Headscale)

El SSH a la Nano y el dashboard van por **Tailscale**, no por SSH directo capa-3. La WiFi de UAO (FortiGate) interfiere con Tailscale.

### Diagnóstico empírico in-situ (2026-05-23, Wi-Fi UAO)

| Hipótesis | Veredicto | Evidencia |
|---|---|---|
| H1: DNS filtering `*.tailscale.com` | ❌ Descartada | DNS resuelve a IPs canónicas (`192.200.0.x`) |
| H2: **FortiGate App-ID categoriza Tailscale** | ✅ **Confirmada** | Cliente reporta health warning `Fortinet may be blocking Tailscale traffic` |
| H3: UDP outbound deny-all | ❌ Descartada | `netcheck` → `UDP: true` |
| H4: **Symmetric NAT corporativo** | ✅ **Confirmada** | `MappingVariesByDestIP: true` |

**Conclusión:** UAO hace **bloqueo selectivo en la fase de autenticación** (App-ID Fortinet sobre `controlplane.tailscale.com`), no bloqueo total. Combinado con symmetric NAT, fuerza modo **DERP-only** pero no impide el túnel si se autentica fuera de la red. `tailscale ping nano` → `pong via DERP(mia) in 122-160ms`.

### Workaround empírico (sin infraestructura adicional)

**Bypass por orden de conexión:** una vez establecida la sesión TCP/443 al DERP relay, FortiGate ya no inspecciona el contenido (WireGuard cifrado encapsulado en TLS, se ve como HTTPS genérico).

1. Antes de entrar a UAO (o al cambiar de red): conectar el laptop a datos móviles 4G.
2. `tailscale up` y completar cualquier auth pendiente en el browser.
3. Verificar con `tailscale status` que la Nano aparezca `active`.
4. Conmutar a `WiFi-UAO`. La sesión sobrevive porque solo necesita TCP/443 outbound.
5. Operar normal: SSH, dashboard `ws://<ip-tailnet>:8000/ws`. Si cae offline, repetir desde el paso 1.

**Dashboard validado en UAO con el workaround:** 12 FPS (target 14), 86 ms/frame de latencia de procesamiento, Nano a 30 °C, detectando "plástico 63%".

### Solución definitiva — Headscale self-hosted (deploy 2026-05-23)

Reemplaza el control plane de Tailscale Inc. (bloqueado) por un Headscale propio bajo un dominio que FortiGate **no cataloga como VPN**. Control plane y DERP viven en el **mismo hostname/puerto 443** que ya validamos que atraviesa el FortiGate.

| Componente | Tailscale Inc. (bloqueado) | Headscale (este deploy) |
|---|---|---|
| Control plane | `controlplane.tailscale.com` → categoría VPN → **bloqueado** | `80-241-217-130.nip.io` → **pasa** |
| DERP (relay) | `derp*.tailscale.com` (fallan incluso fuera de UAO) | **DERP embebido en el VPS** (region 999), 443 + STUN 3478 |

**Infraestructura:** VPS Contabo (`80.241.217.130`, Ubuntu 24.04). Headscale v0.28.0 + Caddy (TLS automático Let's Encrypt) en Docker, aislados en `/opt/headscale/`. **El sitio Frevalle preexistente (nginx :80) NO se tocó** — el puerto 443 estaba libre. UFW abre adicionalmente: `443/tcp+udp` (Caddy + DERP), `41641/udp` (WireGuard), `3478/udp` (STUN).

**Nodos del tailnet `embebidos3`:** vps-frevalle `100.64.0.1` · nano-jetson `100.64.0.2` · laptop `100.64.0.3`.

**Credenciales:**
- Control plane: `https://80-241-217-130.nip.io` (`/health` → `{"status":"pass"}`).
- Pre-auth key reusable (válida 1 año, en `/opt/headscale/PREAUTH_KEY.txt`):
  `hskey-auth-d4BQkuS34qpD-08wvIo3Jy5R2Dqb0UHZxL_WhqPEKIGJa_r2BbW2Eya3-0N9FMxVfEHIHvf6mGj84`
- SSH a la Nano: ahora por llave (`id_ed25519`), ya no depende de Tailscale SSH.

**Conectar el laptop (desde UAO o cualquier red):**
```powershell
tailscale up --login-server=https://80-241-217-130.nip.io --authkey=hskey-auth-d4BQkuS34qpD-08wvIo3Jy5R2Dqb0UHZxL_WhqPEKIGJa_r2BbW2Eya3-0N9FMxVfEHIHvf6mGj84 --accept-dns=false
ssh -i ~/.ssh/id_ed25519 jetson@100.64.0.2
```

**Estado de validación:** Fases 1-5 ✅ (Headscale + Caddy desplegados, VPS + Nano + laptop unidos, DERP embebido activo, SSH OK, Frevalle intacto). En casa el camino laptop↔Nano es **directo** (UDP 41641). **Pendiente: validación física desde UAO** — donde FortiGate forzará el DERP embebido sobre `nip.io:443`, que es justo el componente diseñado para atravesar el firewall.

**Rollback total:**
```bash
cd /opt/headscale && docker compose down -v && rm -rf /opt/headscale
ufw delete allow 41641/udp && ufw delete allow 3478/udp
```

### Negative constraints (acceso remoto)

- Cambiar de control plane exige **`--force-reauth`** (no basta `--reset`).
- **NO** tocar el sitio Frevalle del VPS (nginx en :80). Headscale es aditivo.
- Migración de la Nano debe hacerse con **dead-man-switch** (`systemd-run --on-active=12min` que restaura Tailscale Inc. si no se confirma éxito) — ya consumido, la Nano es estable.
- Plan B garantizado para la sustentación: **hotspot 4G** (bypasa toda la red UAO).

---

## Referencias clave (selección)

| Tema | Fuente |
|---|---|
| SAHI (fundacional, +6,8 pts AP50 solo inferencia) | Akyon, Altinuc, Temizel 2022 — [arXiv:2202.06934](https://arxiv.org/abs/2202.06934) |
| Comparativa YOLO small objects (no migrar) | Tariq, Javed 2025 — [arXiv:2504.09900](https://arxiv.org/abs/2504.09900) |
| P2 head (future work IEEE) | Khalili, Smyth 2024 — [arXiv:2408.04786](https://arxiv.org/abs/2408.04786) |
| YOLOv10 bloqueado en TRT 8.2 | THU-MIG/yolov10 [#75](https://github.com/THU-MIG/yolov10/issues/75), [#129](https://github.com/THU-MIG/yolov10/issues/129) |
| SAHI + TRT engine (torch-free) | obss/sahi [PR #1336](https://github.com/obss/sahi/pull/1336), [PR #1046](https://github.com/obss/sahi/pull/1046) |
| Vast.ai SDK | [vast-ai/vast-python](https://github.com/vast-ai/vast-python) |
| FortiGate bloquea Tailscale (caso idéntico) | tailscale [#11789](https://github.com/tailscale/tailscale/issues/11789), [#15217](https://github.com/tailscale/tailscale/issues/15217) |
| Headscale self-hosted | [juanfont/headscale](https://github.com/juanfont/headscale) · [Janhouse — DERP propio](https://www.janhouse.lv/blog/network/self-hosting-tailscale-derp-headscale) |

> Las listas completas de fuentes (≈200 URLs: papers de waste detection, issues de Ultralytics, catálogos de YouTube, foros) están en los 4 documentos originales.
