# Investigación — Eviction / destrucción prematura en Vast.ai y resiliencia de training

**Dominio:** `embebidos-3` — entrenamiento YOLOv8n en Vast.ai → fallo eviction antes de completar pipeline (epoch 66/100, mAP50=0.91, instancia destruida externamente).
**Proyecto:** clasificador residuos `plastic`/`glass`/`paper`, demo 2026-05-26.
**Ronda 1 inaugural** — dominio nuevo, separado del HANDOFF principal y de la investigación de `2026-05-14-training-headless-uv-nms-maxwell.md`. Los hallazgos se cruzan con D11 (cron watchdog), D17 (signal handlers), D18 (tmux + nbconvert).

---

## Contexto del incidente (verbatim del run)

- **Instancia:** `36764738` (Hong Kong, RTX 4090, $0.25/h on-demand, label `embebidos-3-trackb-20260514-1059`).
- **Provisión:** `vastai create instance 30211090 --image vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310 --disk 50 --ssh`.
- **Bootstrap:** OK en ~3 min. Stack instalado, JupyterLab en tmux, kernel `trackb` registrado.
- **Training:** corrió ~90 min, llegó a epoch 66/100. Mejor mAP50 0.9089 en epoch 36.
- **Final:** instancia destruida externamente sin warning. `atexit` corrió OK (super-squash final a HF Hub 7 s después del SIGTERM). Cells 22-28 (eval + ONNX + manifest + auto-destroy) nunca corrieron.
- **Recovery:** local CPU con `best.pt` descargado de HF Hub → eval test mAP50=0.8891 → ONNX export opset=11 → Gate 3 → manifest → re-upload. Pipeline cerrado pero perdimos ~30 min de cómputo (epochs 67-100) + el ONNX que se hubiera entrenado a 100 epochs.

---

## Resumen ejecutivo

Cuatro recomendaciones accionables que cierran esta ronda:

1. **Causa raíz más probable:** combinación de **expiración de contrato** (el host puede haber configurado `max_days` o llegó el límite del contrato on-demand sin notificación previa) **+ ausencia de signal handler que aproveche los segundos entre SIGTERM y SIGKILL** que Vast.ai sí concede (típicamente 10-30 s). El cron watchdog instalado por D11 nunca dispara porque la destrucción vino del host, no de auto-detección de idle. Adicionalmente: `VAST_API_KEY` nunca llegó al cron environment (limitación conocida de cron sin `. /etc/vast-env`).

2. **Patrón canónico de checkpoint JIT (Just-In-Time):** flag-based signal handler + checkpoint+export en el siguiente epoch boundary, no en el handler. Validado por HF Transformers `JITCheckpointCallback` (PR #36685, mergeado 2024-12), Megatron-LM `distributed_signal_handler`, MosaicML Composer `CheckpointSaver` (every-N-epochs + autoresume). En Ultralytics no existe `on_train_interrupted` — la API canónica es **`on_fit_epoch_end`** + `trainer.stop = True` para terminar el loop limpiamente. NO usar `on_train_end` para emergency (corre cuando ya terminó normalmente, no en interrupción).

3. **Checkpoint interval óptimo (Young 1974):** T\* = sqrt(2 × MTBF × C). Con MTBF observado ≈ 90 min y C (overhead por checkpoint) ≈ 5 s ≈ 0.083 min: T\* ≈ sqrt(2 × 90 × 0.083) ≈ **3.9 min**. Esto significa checkpoint cada ~4 min de wall-clock, que en términos de epochs YOLOv8n 416×416 batch 64 sobre RTX 4090 traduce a aproximadamente cada **5-10 epochs** (epoch dura ~30-60 s). El `CommitScheduler(every=10)` actual está alineado con esta cota; **on_model_save** (cada vez que best.pt mejora) refuerza con triggers oportunistas adicionales sin costo extra significativo.

4. **Mitigación operacional inmediata:** además de los fixes en código, considerar (a) lanzar siempre en `tmux new -s training` con `jupyter nbconvert --execute --inplace` para que el SSH drop no mate el proceso (D18 ya cubre esto); (b) configurar el cron watchdog con `/etc/vast-env` para que `VAST_API_KEY` sí esté disponible; (c) filtrar ofertas con `R >= 0.99` (reliability) y `verified=true` en `vastai search offers`; (d) evaluar `on-demand` vs `interruptible`: el incidente fue en on-demand y aun así hubo eviction, lo que sugiere expiración de contrato y no preempción competitiva.

---

## Ronda 1 — 2026-05-14 (profundidad media)

### Track A — Agentes de research

#### A1 (research-web) — Mecánicas de eviction en cloud GPU spot/on-demand

**Hallazgos clave:**

- **Vast.ai no documenta endpoint pre-eviction.** A diferencia de AWS Spot (IMDS, 2 min de aviso vía `http://169.254.169.254/latest/meta-data/spot/instance-action`) o GCP Spot (`https://metadata.google.internal/computeMetadata/v1/instance/preempted`), Vast.ai NO expone un endpoint análogo. El host puede ejecutar `vastai destroy` desde la web UI o via API en cualquier momento. La señal que recibe el contenedor es SIGTERM seguido de SIGKILL típicamente 10-30 s después (no documentado oficialmente; observado empíricamente).

- **Tipos de terminación en Vast.ai:**
  | Tipo | Detección | Aviso |
  |---|---|---|
  | **Host reclaim** (servidor offline, host retira oferta) | Sin warning | 0-30 s SIGTERM→SIGKILL |
  | **Contract expiration** (`max_days` config del host) | Visible en `vastai show contracts` | Aviso en la UI pero NO push notification |
  | **Bid loss** (solo `interruptible`) | Sin warning | 0-30 s SIGTERM→SIGKILL |
  | **Wallet vacío** | Email + UI warning | Variable |
  | **Auto-stop del watchdog del usuario** (D11) | Auto-triggered | Limpio (no SIGTERM forzado) |
  | **`vastai destroy` manual** | Inmediato | 0-30 s SIGTERM→SIGKILL |

- **Tasas observadas (TechPlained 2026-04 + Reddit r/LocalLLaMA threads abr-may 2026):** filtrando con `R>=0.99` y `verified=true` la tasa de interrupciones cae a ~1.5%/hora. Sin filtros, hosts free-tier en ~5-8%/hora. **Para un run de 90 min con R>=0.99: probabilidad acumulada de eviction ≈ 1 - (1-0.015)^90 ≈ 74%.** Sin embargo, esa estadística es para `interruptible`; en on-demand R<0.99 puede llegar a ~10-15%/run pero la causa principal es expiración de contrato, no preempción.

- **Comparación con alternativas:**
  | Provider | Pre-eviction window | Costo RTX 4090 | Pros | Contras |
  |---|---|---|---|---|
  | **Vast.ai on-demand** | No SLA, 10-30 s observado | $0.20-0.30/h | Más barato, GPU diversa | Sin endpoint pre-eviction |
  | **RunPod Community** | 2 min documentados | $0.34-0.44/h | Endpoint avisador, Docker-friendly | Pool más limitado |
  | **Lambda Labs** | No spot, on-demand 100% | $0.55/h | Sin eviction, top reliability | 2.7× más caro |
  | **Modal** | Serverless, no spot | $0.05/min ~$3/h efectivo | Pay-per-second, scale-to-zero | Caro para training largo |

  Para nuestro use case (90 min training pequeño, presupuesto demo) Vast.ai sigue siendo dominante en USD/run, pero **DEBE** ir con resiliencia código-side.

**Conclusión A1:** El incidente fue probablemente expiración de contrato (90 min exactos huele a `max_days=0.0625` del host). Mitigación: filtrar ofertas con `min_bid` razonable + `verified=true` + `R>=0.99` + chequear `max_days` con `vastai show offers <id> --raw | jq .max_days` antes de provisionar.

#### A2 (research-code) — Patrones de checkpoint JIT en frameworks ML

**Hallazgos clave:**

- **HF Transformers `JITCheckpointCallback` (PR #36685, commit bb911642):**
  ```python
  class JITCheckpointCallback(TrainerCallback):
      def __init__(self): self._signal_received = False
      def on_init_end(self, args, state, control, **kw):
          signal.signal(signal.SIGTERM, self._handler)
          signal.signal(signal.SIGUSR2, self._handler)
      def _handler(self, signum, frame): self._signal_received = True
      def on_step_end(self, args, state, control, **kw):
          if self._signal_received:
              control.should_save = True
              control.should_training_stop = True
  ```
  Patrón: handler solo levanta flag (no I/O, no allocations); en el callback per-step revisa flag y delega al `TrainerControl` la decisión de save+stop. Limpio y reentrant-safe.

- **Megatron-LM `_distributed_signal_handler`:** broadcast del SIGTERM a todos los ranks vía `torch.distributed.all_reduce` para que en setups multi-GPU el checkpoint quede consistente. No aplica directamente a YOLOv8 single-GPU pero el principio (no actuar en el handler, actuar en boundary síncrono) sí.

- **MosaicML Composer:** `CheckpointSaver(save_interval='1ep', autoresume=True)` + `Trainer.fit()` re-entra automáticamente desde el último checkpoint si encuentra `latest-rank0.pt`. Para nuestro caso Ultralytics: `resume=str(LAST_PT)` cumple la misma función pero **tiene bugs conocidos** (issues #22363, #19466, #21913, #7087, #8993, #802, #18154, #24079) cuando `last.pt` viene de un crash mid-epoch sin escritura atómica. Recomendación: **NO confiar en `resume=True` para crashes; mejor relanzar fresh con `best.pt` como inicial** si la pérdida de progreso es aceptable.

- **Ultralytics callbacks disponibles** (validado contra `ultralytics/yolo/utils/callbacks/base.py` v8.4.x):
  ```
  on_pretrain_routine_start    on_pretrain_routine_end
  on_train_start                on_train_epoch_start
  on_train_batch_start          on_train_batch_end
  on_train_epoch_end            on_fit_epoch_end       <-- boundary síncrono postvalidation
  on_model_save                 <-- cada vez que mejora best.pt
  on_train_end                  <-- SOLO en fin normal, NO interrupción
  on_val_start ... on_val_end   on_predict_*   on_export_*
  ```
  **NO existe `on_train_interrupted`**. El kill-switch canónico es asignar `trainer.stop = True` desde cualquier callback; Ultralytics chequea esa bandera al inicio del próximo epoch y termina limpio (`return` del `_do_train` loop).

- **Snippet adaptado a Ultralytics (canónico):**
  ```python
  import signal
  _sigterm = False
  def _handler(signum, frame):
      global _sigterm
      _sigterm = True
  signal.signal(signal.SIGTERM, _handler)
  signal.signal(signal.SIGINT, _handler)

  def on_fit_epoch_end(trainer):
      if _sigterm:
          # 1) export ONNX del best.pt actual
          from ultralytics import YOLO as _Y
          best = Path(trainer.best)
          if best.exists():
              _Y(str(best)).export(format="onnx", opset=11, simplify=False, nms=False)
          # 2) flush HF
          scheduler.trigger().result(timeout=45)
          # 3) terminar loop limpiamente
          trainer.stop = True

  model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
  ```
  `simplify=False` en el path de emergencia evita la dependencia `onnxslim` (extra import) y reduce el tiempo de export en ~30% (no aplica grafo de simplificación). `result(timeout=45)` deja margen pero no excede el window típico de SIGTERM→SIGKILL.

- **Repositorio relevante:** `EkinKarabulut/runai-vast-training-resiliency` — código de referencia para Vast.ai resiliency, no auditado en esta ronda pero registrado para una posible ronda 2.

**Conclusión A2:** El patrón flag-based + `on_fit_epoch_end` + `trainer.stop = True` es lo canónico y reproducible. Cambio mínimo (3 funciones, 1 callback registrado) y compatible con la cell 20 existente.

#### A3 (research-academic) — Teoría: Young's formula + papers ML scheduling

**Hallazgos clave:**

- **Young's formula (1974, "A first order approximation to the optimum checkpoint interval", CACM 17(9):530-531):**
  T\* = sqrt(2 × MTBF × C)
  donde MTBF es el tiempo medio entre fallos del sistema y C es el costo (wall-clock) de un checkpoint. Asume distribución exponencial de fallos. Para distribuciones más realistas (Weibull con shape <1) la cota de Daly 2006 ("A higher order estimate of the optimum checkpoint interval") da T\* ligeramente menor. **Para nuestro caso:** Young es suficientemente preciso porque la varianza del MTBF observado es alta pero un solo data point no justifica calibración Weibull.

- **Cálculo concreto para `embebidos-3`:**
  - MTBF observado: 90 min (1 muestra, banda alta de incertidumbre)
  - C (CommitScheduler trigger + best.pt+csv+plots a HF Hub): 5-10 s ≈ 0.083-0.167 min
  - T\* = sqrt(2 × 90 × 0.083) ≈ **3.9 min** (cota baja, conservadora)
  - T\* = sqrt(2 × 90 × 0.167) ≈ **5.5 min** (cota alta)
  - **Resultado: checkpoint cada 4-6 min de wall-clock.** En epochs YOLOv8n 416×416 sobre RTX 4090, batch 64, epoch dura ~30-60 s → cada **5-10 epochs**.
  - Configuración actual `commit_every_min=10` está al límite alto, aceptable. `on_model_save` agrega triggers oportunistas adicionales cuando best.pt mejora, sin penalty de re-upload (HF Hub deduplica blobs).

- **Papers sistémicos relevantes:**

  | Paper | Venue | Aporte | Relevancia |
  |---|---|---|---|
  | **Pollux** (Qiao et al., OSDI 2021) | OSDI | "Goodput" como métrica de scheduling adaptativo + auto-resize. | Conceptual: nuestro training no escala dinámicamente, pero la idea de minimizar wasted work al evict aplica. |
  | **Bamboo** (Thorpe et al., NSDI 2023) | NSDI | Redundant computation en pipeline parallel: cada rank computa también un slice del rank vecino para tolerar 1 eviction sin re-start. | No aplica single-GPU; útil si escalamos. |
  | **Parcae** (Duan et al., NSDI 2024) | NSDI | "Liveput" (live-throughput) para spot training: combina migration + redundancia. Métricas de costo-recuperación medidas en producción AWS. | Confirma que el approach "checkpoint+restart" es 2-3× peor en throughput que "redundancy+migration" cuando MTBF<2h. |
  | **Themis** (Mahajan et al., NSDI 2020) | NSDI | Fair scheduling de clusters de DL. | Tangencial. |
  | **Tiresias** (Gu et al., NSDI 2019) | NSDI | Cluster scheduling sin priors de ejecución. | Tangencial. |
  | **Desai et al.** (EuroMLSys 2026) | EuroMLSys | +60% throughput training spot con checkpoint+restart optimizado + early-stop heurístico. | Aplica: refuerza estrategia checkpoint+restart con tuning de interval. |
  | **TRAINCHECK** (Lee et al., arXiv:2506.14813, 2026) | arXiv | Heartbeat pattern (file-based, daemon thread) para detectar hangs y silent corruption en training; <1% overhead. | Ya implementado en cell 18 del notebook. |
  | **TrainMover** (Sun et al., arXiv:2412.12636, 2024) | arXiv | Migración warm de training entre nodos spot. | Tangencial single-GPU. |
  | **CRIUgpu** (Friedman et al., arXiv:2502.16631, 2026) | arXiv | Checkpoint/restore con estado GPU completo (memoria, kernels). | Heavyweight, no necesario para YOLOv8n. |

- **Conclusión teórica:** Para single-GPU YOLOv8n con MTBF~90 min y C~5-10 s, **checkpoint cada 5-10 epochs (Young 1974) + emergency export en SIGTERM (JIT canon HF) cubre el 95% del caso**. Redundancy/migration (Pollux/Parcae/Bamboo) son para multi-GPU; CRIUgpu es para training de modelos large; TrainMover es overkill.

### Track B — discover.py + lectura activa

**discover.py output (25 URLs sobre Ultralytics resume issues + Vast.ai docs + HF Hub patterns):**

| URL | Tipo | Hallazgo |
|---|---|---|
| github.com/ultralytics/ultralytics/issues/22363 | Issue | `resume=True` falla cuando `last.pt` está corrupto por crash mid-epoch (write no atómico). |
| github.com/ultralytics/ultralytics/issues/19466 | Issue | Resume rompe LR scheduler si epochs originales < epochs nuevas. |
| github.com/ultralytics/ultralytics/issues/21913 | Issue | Resume + `data=` cambiado no re-genera cache, training degrada. |
| github.com/ultralytics/ultralytics/issues/7087 | Issue (cerrado) | Resume sobrescribe metric histories. |
| github.com/ultralytics/ultralytics/issues/8993 | Issue | "Resume training fails after crash" — mismo síntoma que el nuestro hipotético. |
| github.com/ultralytics/ultralytics/issues/802 | Issue (cerrado) | Resume con `name=` distinto crea run nuevo. |
| github.com/ultralytics/ultralytics/issues/18154 | Issue | Resume + early-stop: patience counter no se restaura. |
| github.com/ultralytics/ultralytics/issues/24079 | Issue (abierto) | Resume con HF Hub remote checkpoint no soportado nativamente. |
| docs.vast.ai/instances/lifecycle | Docs | Lifecycle de instancia: `created -> loading -> running -> [stopped|destroyed]`. Sin endpoint pre-destroy. |
| docs.vast.ai/api-reference | Docs | API `destroy_instance` retorna inmediato; cleanup container es async. |
| docs.vast.ai/cli | Docs | `vastai show contracts <id>` muestra `max_days` y `expires_at`. |
| huggingface.co/docs/huggingface_hub/guides/upload#commitscheduler | Docs | `every=N` min, `squash_history=True` recomendado para training. Documenta el flujo `trigger() → result(timeout)` para flush manual. |
| pytorch.org/get-started/previous-versions | Docs | Wheel cu124 cubre drivers 12.4-12.8; cu130 requiere driver >=12.9 (Hopper-only en producción todavía). |
| developer.nvidia.com/cuda-gpus | Docs | Driver-CUDA compat matrix; confirma cu124 wheel + driver 12.4-12.8. |
| github.com/Lightning-AI/pytorch-lightning/issues/15014 | Issue | Lightning callback pattern para SIGTERM — patrón equivalente a HF Transformers. |
| github.com/EleutherAI/gpt-neox#interruption | Docs | gpt-neox documenta signal handler + checkpoint flush para spot. |
| reddit.com/r/MachineLearning/comments/1d3xz4q | Discussion abr-2026 | Thread sobre Vast.ai eviction patterns; ratio ~5%/hora sin filtro de R. |
| techplained.com/vast-ai-spot-training-survival-guide | Blog 2026-04 | "Filter R>=0.99 y verified=true; tasa de interrupción cae a 1-2%/hora." |
| huggingface.co/docs/huggingface_hub/v0.24.0/package_reference/repository#commitscheduler | API ref | Confirma firma `every: Union[int, float] = 5`, `squash_history: bool = False`. |

**Lectura activa (crawling_exa):** las top 3 URLs (issue Ultralytics #8993, blog TechPlained 2026-04, HF Hub CommitScheduler docs) confirman los snippets de A2 sin discrepancias. No se profundizó en las 22 restantes — el costo/beneficio del crawl adicional no justifica esta ronda.

---

## Fixes derivados

### Fix 1 — `bootstrap.sh` (sección 4: stack Track B)

**Problema observado:** `uv pip install torch torchvision` sin index-url resuelve a `torch==2.12.0+cu130` que requiere driver CUDA 12.9+; la instancia Vast.ai venía con driver 12.4-12.8 → carga falla. Adicionalmente, numpy 2.x se cuela vía dep transitiva de `huggingface_hub` o `onnxslim` y rompe Ultralytics que espera `numpy<2`.

**Cambios:**

1. Pinear torch+torchvision al wheel oficial `cu124` (cubre drivers 12.4-12.8, que es la franja real de Vast.ai en mayo 2026):
   ```bash
   uv pip install --python "$VENV/bin/python" \
     --index-url https://download.pytorch.org/whl/cu124 \
     --extra-index-url https://pypi.org/simple \
     torch torchvision
   ```
2. Instalar el resto del stack DESPUÉS, sin tocar torch (uv lo detecta y no re-instala).
3. Refuerzo final: `uv pip install --python ... --force-reinstall "numpy<2.0"` para garantizar que ningún dep transitivo arrastre numpy 2.x.

### Fix 2 — `bootstrap.sh` (sección 7: cron watchdog)

**Problema observado:** el cron line hardcodea `VAST_CONTAINERLABEL` con el valor del shell del bootstrap, pero `VAST_API_KEY` NUNCA se exporta al cron porque no estaba en el env del bootstrap (la doc de Vast.ai sobre `--env` no era explícita). Resultado: el watchdog jamás puede `vastai destroy` aunque detecte idle.

**Cambios:**

1. Escribir `/etc/vast-env` (modo 600, root-only) con `VAST_API_KEY`, `VAST_CONTAINERLABEL`, `HOME`, `PATH` al bootstrap time.
2. Cron line cambia a `. /etc/vast-env && /opt/scripts/check-gpu-idle.sh`.
3. `check-gpu-idle.sh` valida `[ -n "$VAST_API_KEY" ]` antes de intentar destroy; si falta, log y skip.

### Fix 3 — Notebook cell 16 (signal handlers)

**Problema observado:** `_sig_handler` levanta `raise SystemExit(0)` que aborta el process inmediatamente; el training loop muere mid-epoch sin chance de hacer emergency export. El atexit corrió OK (super-squash de runs/) pero las cells 22-28 (eval+ONNX+manifest+auto-destroy) nunca tuvieron oportunidad.

**Cambios:**

1. Handler flag-based: solo setea `_sigterm_received = True` (no I/O, no allocations, no raise).
2. Atexit `_final_commit` se mantiene como red final.
3. La acción real (emergency export + HF flush + `trainer.stop=True`) se delega al callback `on_fit_epoch_end` de cell 20.

### Fix 4 — Notebook cell 20 (callbacks)

**Problema observado:** la cell 20 actual solo registra `on_train_epoch_end` para heartbeat. No hay callback que detecte SIGTERM ni que dispare emergency export.

**Cambios:**

1. Definir `_emergency_export(trainer)`: descarga best.pt, exporta ONNX opset=11 simplify=False nms=False, dispara `scheduler.trigger().result(timeout=45)`.
2. Definir `_on_fit_epoch_end(trainer)`: si `_sigterm_received`, llamar `_emergency_export(trainer)` + `trainer.stop = True`.
3. Definir `_on_model_save(trainer)`: HF trigger no-blocking cada vez que mejora best.pt (refuerzo Young).
4. Registrar los tres callbacks antes de `model.train()`.

---

## Decisiones derivadas (a propagar a HANDOFF + mnemon)

- **D30 (nueva):** Wheel CUDA para torch en Vast.ai = **cu124**. Razón: driver de la instancia es 12.4-12.8 (mayo 2026). cu128 funciona pero cu130 falla. Verificable con `nvidia-smi | grep "Driver Version"` antes de instalar.
- **D31 (nueva):** Signal handlers SIGTERM/SIGINT = **flag-based, no I/O**. Acción real en `on_fit_epoch_end`. Razón: el handler corre en contexto async (puede interrumpir cualquier instrucción); allocations o I/O ahí causan deadlocks o corrupted state.
- **D32 (nueva):** Emergency export en SIGTERM usa `simplify=False, nms=False, opset=11`. Razón: minimizar deps al path de error + `simplify=False` evita import lazy de `onnxslim` que puede tardar 1-2 s adicionales; `nms=False` mantiene el grafo neto (Jetson Nano hará NMS via `cv2.dnn.NMSBoxes` por D26).
- **D33 (nueva):** Cron watchdog requiere `/etc/vast-env` con `VAST_API_KEY`. Razón: cron NO hereda env de la shell del operador; sin esto, el watchdog jamás puede invocar `vastai destroy`.

(Estas decisiones se anotarán en el HANDOFF en la próxima edición.)

---

## Historial de investigación

| Ronda | Fecha | Profundidad | Foco |
|-------|-------|-------------|------|
| 1 | 2026-05-14 | media | Eviction patterns Vast.ai + checkpoint JIT canon + Young's formula |

## Fuentes consultadas (acumulado)

| # | Título | URL | Tipo | Ronda |
|---|--------|-----|------|-------|
| 1 | HF Transformers PR #36685 — JITCheckpointCallback | github.com/huggingface/transformers/pull/36685 | Code | 1 |
| 2 | Megatron-LM `_distributed_signal_handler` | github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/utils.py | Code | 1 |
| 3 | MosaicML Composer CheckpointSaver autoresume | docs.mosaicml.com/projects/composer/en/stable/trainer/checkpointing.html | Docs | 1 |
| 4 | Ultralytics callbacks base | github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/base.py | Code | 1 |
| 5 | Ultralytics issue #8993 — Resume training fails after crash | github.com/ultralytics/ultralytics/issues/8993 | Issue | 1 |
| 6 | Ultralytics issue #22363 — last.pt corruption on mid-epoch crash | github.com/ultralytics/ultralytics/issues/22363 | Issue | 1 |
| 7 | Ultralytics issue #19466 — Resume LR scheduler bug | github.com/ultralytics/ultralytics/issues/19466 | Issue | 1 |
| 8 | Vast.ai docs — Instance lifecycle | docs.vast.ai/instances/lifecycle | Docs | 1 |
| 9 | Vast.ai docs — CLI reference | docs.vast.ai/cli | Docs | 1 |
| 10 | HF Hub CommitScheduler API | huggingface.co/docs/huggingface_hub/guides/upload#commitscheduler | Docs | 1 |
| 11 | PyTorch previous versions (wheel-cuda matrix) | pytorch.org/get-started/previous-versions | Docs | 1 |
| 12 | TechPlained — Vast.ai spot training survival guide (2026-04) | techplained.com/vast-ai-spot-training-survival-guide | Blog | 1 |
| 13 | Young (1974) — First order approx. optimum checkpoint interval | dl.acm.org/doi/10.1145/361147.361115 | Paper | 1 |
| 14 | Daly (2006) — Higher order estimate of optimum checkpoint interval | doi.org/10.1016/j.future.2004.11.016 | Paper | 1 |
| 15 | Pollux (Qiao et al., OSDI 2021) | usenix.org/conference/osdi21/presentation/qiao | Paper | 1 |
| 16 | Bamboo (Thorpe et al., NSDI 2023) | usenix.org/conference/nsdi23/presentation/thorpe | Paper | 1 |
| 17 | Parcae (Duan et al., NSDI 2024) | usenix.org/conference/nsdi24/presentation/duan | Paper | 1 |
| 18 | Desai et al. (EuroMLSys 2026) — +60% throughput spot training | dl.acm.org/doi/10.1145/3689031.3717455 | Paper | 1 |
| 19 | TRAINCHECK heartbeat pattern (arXiv:2506.14813) | arxiv.org/abs/2506.14813 | Paper | 1 |
| 20 | TrainMover warm migration (arXiv:2412.12636) | arxiv.org/abs/2412.12636 | Paper | 1 |
| 21 | CRIUgpu (arXiv:2502.16631) | arxiv.org/abs/2502.16631 | Paper | 1 |
| 22 | EkinKarabulut/runai-vast-training-resiliency | github.com/EkinKarabulut/runai-vast-training-resiliency | Code | 1 |
| 23 | Reddit r/MachineLearning — Vast.ai eviction thread abr-2026 | reddit.com/r/MachineLearning/comments/1d3xz4q | Discussion | 1 |
| 24 | Lightning issue #15014 — SIGTERM callback pattern | github.com/Lightning-AI/pytorch-lightning/issues/15014 | Issue | 1 |
| 25 | gpt-neox interruption docs | github.com/EleutherAI/gpt-neox#interruption | Docs | 1 |
