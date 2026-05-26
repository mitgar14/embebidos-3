# Automatización del ciclo de entrenamiento en Vast.ai (embebidos-3)

Objetivo: hacer robusto y automático todo el ciclo de entrenamiento del modelo
(YOLOv8n 4 clases) en Vast.ai. Los cuatro requisitos del usuario:

1. Script que busque ofertas e importe credenciales desde `.env`.
2. Crear instancia garantizando compatibilidad con el stack que importamos.
3. Persistir logs y modelo en HF Hub e integrarlo con la plataforma web (SolidJS) para monitoreo en vivo.
4. Al terminar: persistir todo y apagar o destruir la instancia de forma fiable.

---

## Ronda 1 — 2026-05-25 (profundidad media)

### Contexto e infraestructura actual

Lo que ya existe (no partimos de cero):

- `scripts/training/onstart.sh`: lo que Vast.ai corre vía `--onstart`. Exige `HF_TOKEN`, descarga `bootstrap.sh` + el notebook desde HF (`mitgar14/embebidos3-raw-batches/.notebook/`) y lanza `nbconvert` dentro de `tmux`.
- `scripts/training/bootstrap.sh`: instala el stack en `/opt/venv/trackb`, registra el kernel `trackb`, monta un cron watchdog auto-destroy por GPU idle.
- `notebooks/train_v1d_vastai.ipynb`: `huggingface_hub.CommitScheduler` empuja `runs/`, `manifests/`, `exports/` a HF cada 10 min; `heartbeat.jsonl` (epoch, loss, grad_norm, lr, gpu_mem, eta cada 30s); signal handlers SIGTERM con commit final; y `vastai destroy` al final del notebook.
- `.env` (gitignoreado): `HF_TOKEN`, `VAST_API_KEY`, `ROBOFLOW_API_KEY`.
- Frontend SolidJS en despliegue (Vercel).
- Gotchas validados en notas previas: usar `vastai/base-image` (no `vastai/pytorch`); wheel `cu124` (no `cu130` con driver < 12.9); flags `--ssh --direct --disk` obligatorios.

El punto más frágil del flujo actual: el teardown depende de que el cron watchdog y el `vastai destroy` no fallen en silencio. Si la instancia entra en `exited`/`unknown`/`offline` sin que el watchdog lo note, el billing sigue corriendo.

### Track A — agentes de research

#### A.1 Orquestadores: SkyPilot vs dstack vs raw-CLI (research-code)

**Conclusión: SkyPilot sobre Vast.ai.** Vast.ai es backend nativo y verificado de SkyPilot (provisioning real en `sky/clouds/vast.py` y `sky/provision/vast/utils.py`). dstack también soporta Vast.ai pero con restricciones de su modelo container-based. El raw-CLI sigue siendo el de menor fricción de adopción y mayor control del DSL de búsqueda, a costa de no tener teardown garantizado ni recuperación ante fallas.

Comparación por dimensión (orientada al MVP):

| Dimensión | SkyPilot | dstack | Raw CLI |
|---|---|---|---|
| (a) Búsqueda de oferta | Optimizer elige la más barata que cumpla `accelerators`/`disk`/región (catálogo cacheado) | Delega a `gpuhunt` (abstracción opaca) | DSL completo: `reliability>0.98 cuda_max_good>=12.1 dlperf_usd-` (control total) |
| (b) Compatibilidad CUDA/driver/imagen | `image_id: docker:...`; driver no es campo de primer nivel (pasar vía `create_instance_kwargs`) | Imagen gestionada por dstack; control de CUDA opaco | Filtros directos `cuda_max_good`, `driver_version` (riesgo bajo) |
| (c) Secrets desde `.env` | `secrets: {HF_TOKEN: null}` redactado en logs y dashboard | `env: [HF_TOKEN]` desde el entorno local | `--env "-e HF_TOKEN=..."` (riesgo de filtración en logs) |
| (d) Teardown fiable | `skylet` remoto ejecuta autostop/autodown aunque caiga la laptop; managed jobs limpian al terminar | runner termina el container y destruye la instancia; necesita servidor dstack vivo | cron watchdog + `vastai destroy` (alto riesgo de billing zombie) |
| (e) Streaming de logs | `sky logs --follow` sin SSH manual | `dstack logs` vía WebSocket | `ssh ... tail -f` manual |
| (f) Esfuerzo de adopción | 2 a 4 h; `pip install "skypilot[vast]"` | 4 a 8 h; requiere servidor dstack persistente | 1 a 2 h; ya instalado, pero frágil |

Caveats importantes (de los repos y DeepWiki):

- El catálogo de Vast.ai en SkyPilot está **cacheado**: puede intentar una oferta ya alquilada; el failover no garantiza un reemplazo idéntico.
- SkyPilot en Vast.ai **no soporta**: multi-nodo, montaje de object-store (S3/GCS), abrir puertos después de lanzar (solo al lanzar; se accede por `ssh -L`), tiers de disco/red.
- `sky jobs launch` (managed jobs) levanta un **jobs controller** (cluster pequeño persistente) con costo extra mientras el job corre. Para un run único on-demand, `sky launch --down -i N` es más liviano (sin recuperación automática, pero on-demand no sufre preemption).
- dstack **solo soporta on-demand** en Vast.ai hoy (sin spot/bid) y necesita un servidor de control plane activo durante toda la sesión.
- `vast-ai/vast-sdk` está **deprecado**: absorbido por `vast-ai/vast-cli`. `pip install vastai` instala CLI + SDK (`from vastai import VastAI`).

Repos de referencia: `skypilot-org/skypilot` (`sky/clouds/vast.py`, `sky/provision/vast/utils.py`), `vast-ai/vast-cli`, `elizaOS/eliza` (`packages/training/scripts/cloud/dispatch-vast.sh`, ciclo de vida completo con guards de VRAM), `jeremyadamsfisher/slow_diffusion` (`vastai-create-instance.sh`, inyección de secret vía `--env`).

#### A.2 Puente HF Hub ↔ frontend web (research-web)

Premisa de seguridad: **el token de HF nunca puede vivir en el cliente**. La URL `https://huggingface.co/datasets/{user}/{repo}/resolve/main/{archivo}` acepta `Authorization: Bearer hf_xxx` y devuelve el contenido crudo (401 sin token en repos privados).

**Opción A (recomendada): proxy serverless en Vercel + polling.** `HF_TOKEN` (fine-grained, solo lectura, solo ese repo) vive como env var en Vercel. Una función `api/training-status.ts` hace `fetch` a `resolve/main/heartbeat.jsonl` + `results.csv`, parsea y devuelve JSON; el frontend SolidJS hace polling cada 30s con `setInterval`. CORS no es problema (el fetch lo hace el servidor). Vercel serverless tiene timeout suficiente (request < 2s); SSE solo funciona en Edge runtime (no en serverless Node stateless). `Cache-Control: s-maxage` amortigua llamadas a HF.

**Opción B (recomendada si tocamos el script): trackio + HF Space embebido.** `trackio` (HF, MIT, wandb-compatible) loguea a un HF Space; su dashboard se embebe como `<iframe src="https://org-space.hf.space/?project=...&metrics=loss,epoch&sidebar=hidden">`. Cero backend propio, cero token en el cliente. Caveats: cold start del Space (30 a 60s) y el Space debe ser **público** para que el iframe cargue sin sesión HF.

**Opción C (descartada):** webhook HF + Vercel KV (añade Redis y configuración sin beneficio claro para MVP). **Opción D (válida si la privacidad no importa):** repo público + `fetch` directo desde el browser, sin proxy.

Modelo de seguridad por pieza: `HF_TOKEN` fine-grained read-only en Vercel (nunca al cliente); token de escritura de trackio en Vast.ai (nunca en Vercel); el Space de trackio puede ser público (el iframe no necesita token).

### Track B — búsqueda ampliada (discover.py + YouTube)

`discover.py` (3 queries: automatización Vast.ai, orquestadores, monitoreo HF) + extracción con Exa de las fuentes clave. Hallazgos concretos que sostienen la síntesis:

**SkyPilot + Vast.ai (artículo oficial de Vast, 2026-03-15).**
- Instalación: `pip install -U "skypilot[vast]"` (o `uv pip install "skypilot[vast]"`). En Python 3.12+ hay que instalar `skypilot[vast]` explícito (se excluye de `skypilot[all]`).
- Credenciales: `mkdir -p ~/.config/vastai && echo "<API_KEY>" > ~/.config/vastai/vast_api_key`; verificar con `sky check` (debe decir `Vast: enabled`).
- Ciclo de vida: `sky launch cfg.yaml -c NAME`, `sky logs NAME --follow`, `sky status NAME` (busca `UP`), `sky stop`/`sky start`/`sky down NAME`.
- Config Vast-específica en `~/.sky/config.yaml`: `vast.datacenter_only: true` (excluye GPUs caseras, mejora fiabilidad) y `vast.create_instance_kwargs` (`image`, `onstart_cmd`/`onstart`, `disk`, `env`, `bid_price`, `python_utf8`, `lang_utf8`, `template_hash_id`, `extra: "--shm-size=16g"`).
- Spot: `use_spot: true` + `bid_price`; managed jobs recuperan ante preemption.

**Autostop / Autodown (docs SkyPilot).**
- `resources.autostop: true` (5 min default) | `10m` | `{idle_minutes: 10}`. **Autodown** (destruye, no solo pausa): `autostop: {idle_minutes: 10, down: true}`.
- Lo ejecuta el **cluster remoto** (la laptop no necesita seguir viva).
- `wait_for`: `jobs_and_ssh` (default) | `jobs` | `none` (límite duro de tiempo, ignora jobs largos). Para nuestro caso, `nbconvert` corre como el comando `run:` (un job de SkyPilot), así que el default cuenta la inactividad recién cuando el job termina (no hay que usar `none`).
- **Autostop hooks**: corren un script ANTES del teardown (`hook_timeout` configurable). Un ejemplo de la propia doc es **"Pushing model to Hugging Face Hub"**: `huggingface-cli upload my-org/my-model /workspace/model-output .`. Es el mecanismo nativo de "persistir y luego destruir".

**trackio (`gradio-app/trackio`).** Tracking ligero local-first, SQLite, API compatible con wandb (`import trackio as wandb`). Con `space_id` loguea a un HF Space (gratis). Dashboard embebible vía iframe con query params (`project`, `metrics`, `sidebar=hidden`, `theme`). `log()` es no bloqueante (cola + batch, zero-loss con reintentos), no frena el training. Integra con `transformers.Trainer` (`report_to="trackio"`) y `accelerate`.

**HF Webhooks.** `create_webhook(url=..., watched=[{type,name}], domains=["repo"], secret=...)`; evento `repo.content`/`update` en cada commit. Puede pegarle a una URL o disparar un HF Job (payload en `WEBHOOK_PAYLOAD`). Límite 1.000 disparos/24h (vs ~144 commits/día del scheduler). Requiere un receptor persistente (servidor o Space con `WebhooksServer`), por eso es menos simple que el polling para este MVP.

**dstack + Vast.ai (artículos oficiales de Vast, 2023 y 2026-01).** Integración desde dstack 0.12.3; configurar backend con la API key; quality score + precio máximo; dev environments + tasks + services.

**Sub-track YouTube (canal secundario).** `youtube_search` ubicó charlas de referencia: "Building Multi-Cloud GenAI Platforms Without the Pain" de Romil Bhardwaj, co-creador de SkyPilot (MLOps World 2025, `_oWJ0t-RW7o`); "SkyPilot: Run AI on Any Cloud" (Anyscale, `rgQxO54hN8Q`); keynote "Sky Computing" de Ion Stoica, UC Berkeley (USENIX, `AuNfxVLdo0A`). **Decisión registrada:** se deprioriza la transcripción completa porque las docs oficiales ya aportan el detalle técnico concreto y el tema es doc/código-céntrico; quedan como referencias primarias para profundizar si hace falta.

---

## Síntesis: arquitectura recomendada para embebidos-3

La respuesta "más robusta" no es seguir hilando bash, sino **adoptar SkyPilot como orquestador y conservar lo que ya funciona** (el `CommitScheduler` del notebook como capa de persistencia). SkyPilot cubre los cuatro requisitos de forma nativa; el `CommitScheduler` ya alimenta el monitoreo web sin tocar nada.

```
  Local (laptop)                         Vast.ai (instancia efímera)
  ───────────────                        ────────────────────────────
  sky launch -d -c v1d \                 setup:  (= bootstrap.sh: venv trackb + stack)
    train-v1d.sky.yaml \                 run:    jupyter nbconvert train_v1d_vastai.ipynb
    --down -i 10 \                                └─ CommitScheduler  ──► HF Hub (privado)
    --secret HF_TOKEN=$(.env)                          heartbeat.jsonl / results.csv
        │                                              manifests / exports/best.pt
        │ (skylet remoto, la laptop                          │
        │  puede desconectarse)                              │
        ▼                                                     ▼
  autodown: idle 10 min ──► destruye    HF Hub  ◄── huggingface-cli upload (red final en run:)
                                            │
                                            ▼
  Vercel (frontend SolidJS)
    api/training-status.ts  ── fetch resolve/main/heartbeat.jsonl (HF_TOKEN fine-grained RO)
        │                                              │
        └─ polling 30s ──► panel de métricas      (opcional) iframe trackio.hf.space ──► gráficas
```

Cómo se mapea a cada requisito:

1. **Buscar oferta + importar `.env`:** el optimizer de SkyPilot elige la oferta (elimina el `vastai search offers` manual). `VAST_API_KEY` se escribe una vez en `~/.config/vastai/vast_api_key`; `HF_TOKEN` entra por `--secret HF_TOKEN="$(grep HF_TOKEN .env | cut -d= -f2)"` (redactado por SkyPilot).
2. **Instancia compatible:** `image_id: docker:vastai/base-image:cuda-12.4.1-...` + `vast.datacenter_only: true` (drivers mejor mantenidos). Si el driver/CUDA llegara a morder, `vast.create_instance_kwargs` en `~/.sky/config.yaml`.
3. **Persistir + web:** se **conserva el `CommitScheduler`** (push cada 10 min + commit final en atexit). El monitoreo web reusa `heartbeat.jsonl` vía el **proxy Vercel** (sin tocar el notebook); opcional `trackio` + iframe para gráficas.
4. **Persistir y destruir:** **autodown** (`down: true`) lo ejecuta el `skylet` remoto. Una red final con `huggingface-cli upload best.pt` al cierre del `run:` blinda la persistencia.

### Qué se reemplaza y qué se conserva

| Pieza actual | Bajo SkyPilot |
|---|---|
| `onstart.sh` (descarga + nbconvert + tmux) | Se reemplaza por el `run:` del YAML (`sky logs --follow` sustituye al tmux) |
| `bootstrap.sh` (venv + stack) | Pasa al `setup:` del YAML |
| cron watchdog GPU-idle | **Se elimina** (autodown lo reemplaza, y es más fiable) |
| `vastai destroy` final en el notebook | **Se elimina** (SkyPilot es el dueño del ciclo de vida; mantenerlo duplicaría) |
| `CommitScheduler` + `heartbeat.jsonl` + signal handlers | **Se conserva** (es la capa de persistencia; alimenta el monitoreo web) |

### Camino de migración (incremental)

- Paso 0: `pip install "skypilot[vast]"`, escribir `~/.config/vastai/vast_api_key` desde `.env`, `sky check`.
- Paso 1: escribir `train-v1d.sky.yaml` (traducir `bootstrap.sh` a `setup:`, el `nbconvert` a `run:`). Quitar del notebook el `vastai destroy` y desactivar el cron watchdog.
- Paso 2: `sky launch -d -c v1d train-v1d.sky.yaml --down -i 10 --secret HF_TOKEN=...`. Validar con un run corto (pocas épocas).
- Paso 3: proxy Vercel `api/training-status.ts` + polling en el dashboard SolidJS (reusa `heartbeat.jsonl`, no toca el notebook).
- Paso 4 (opcional): `trackio` en el notebook + iframe del Space para gráficas históricas.

### Decisiones abiertas y riesgos

- **Catálogo cacheado de SkyPilot:** puede elegir una oferta ya alquilada; mitigar con `datacenter_only` y reintentar `sky launch`.
- **Driver/CUDA no es campo de primer nivel** en el YAML de SkyPilot. Si el gotcha cu124/driver muerde, usar `create_instance_kwargs` o caer al raw-CLI solo para la query de oferta.
- **`sky jobs launch` vs `sky launch --down`:** managed jobs dan recuperación pero añaden un controller con costo extra; para un run único on-demand, `sky launch -d --down -i 10` es más liviano y suficiente.
- **trackio Space:** cold start 30 a 60s y debe ser público para el iframe; conviene abrirlo con anticipación antes de la sustentación.
- **Fallback pragmático:** si adoptar SkyPilot consume tiempo, endurecer el bash actual reemplazando el cron watchdog por un loop Python con `from vastai import VastAI` que haga polling de `actual_status` con timeout y destruya la instancia tanto en éxito como en error (snippet en el reporte de research-code).

---

## Historial de investigación

| Ronda | Fecha | Profundidad | Foco |
|-------|-------|-------------|------|
| 1 | 2026-05-25 | Media | Orquestación del ciclo de entrenamiento en Vast.ai: SkyPilot vs dstack vs raw-CLI, puente HF Hub ↔ frontend web, teardown fiable |

## Fuentes consultadas (acumulado)

| # | Título | URL | Tipo | Ronda |
|---|--------|-----|------|-------|
| 1 | Vast.ai GPUs Can Now Be Rented Through SkyPilot | https://vast.ai/article/vast-ai-gpus-can-now-be-rentend-through-skypilot | Artículo oficial | 1 |
| 2 | Autostop and Autodown (SkyPilot Docs) | https://docs.skypilot.co/en/stable/reference/auto-stop.html | Doc oficial | 1 |
| 3 | Managed Jobs (SkyPilot Docs) | https://docs.skypilot.co/en/stable/examples/managed-jobs.html | Doc oficial | 1 |
| 4 | SkyPilot YAML spec | https://skypilot.readthedocs.io/en/stable/reference/yaml-spec.html | Doc oficial | 1 |
| 5 | SkyPilot Python SDK (idle_minutes_to_autostop) | https://skypilot.readthedocs.io/en/latest/reference/api.html | Doc oficial | 1 |
| 6 | skypilot-org/skypilot · sky/clouds/vast.py | https://github.com/skypilot-org/skypilot/blob/master/sky/clouds/vast.py | Código | 1 |
| 7 | skypilot-org/skypilot · sky/provision/vast/utils.py | https://github.com/skypilot-org/skypilot/blob/master/sky/provision/vast/utils.py | Código | 1 |
| 8 | SkyPilot at Shopify (uso en producción) | https://shopify.engineering/skypilot | Ingeniería | 1 |
| 9 | Creating Instances with the API (Vast.ai) | https://docs.vast.ai/api-reference/creating-instances-with-api | Doc oficial | 1 |
| 10 | Vast.ai Python SDK quickstart | https://docs.vast.ai/sdk/python/quickstart | Doc oficial | 1 |
| 11 | vastai create instance (CLI ref) | https://docs.vast.ai/cli/reference/create-instance | Doc oficial | 1 |
| 12 | vast-ai/vast-cli (CLI + SDK unificado) | https://github.com/vast-ai/vast-cli | Código | 1 |
| 13 | vast-ai/vast-sdk (deprecado, redirige a vast-cli) | https://github.com/vast-ai/vast-sdk | Código | 1 |
| 14 | elizaOS/eliza · dispatch-vast.sh | https://github.com/elizaOS/eliza/blob/main/packages/training/scripts/cloud/dispatch-vast.sh | Código | 1 |
| 15 | jeremyadamsfisher/slow_diffusion · vastai-create-instance.sh | https://github.com/jeremyadamsfisher/slow_diffusion/blob/main/vastai-create-instance.sh | Código | 1 |
| 16 | Accessing the GPU marketplace with Vast.ai and dstack | https://vast.ai/article/vastAI-and-dstack | Artículo oficial | 1 |
| 17 | Deploy LLMs with dstack on Vast.ai | https://vast.ai/article/deploy-llms-dstack-vllm-guide | Artículo oficial | 1 |
| 18 | dstackai/dstack | https://github.com/dstackai/dstack | Código | 1 |
| 19 | gradio-app/trackio | https://github.com/gradio-app/trackio | Código | 1 |
| 20 | Introducing Trackio (blog HF) | https://huggingface.co/blog/trackio | Blog oficial | 1 |
| 21 | Trackio docs | https://huggingface.co/docs/trackio/index | Doc oficial | 1 |
| 22 | HF Webhooks (guía huggingface_hub) | https://huggingface.co/docs/huggingface_hub/guides/webhooks | Doc oficial | 1 |
| 23 | HF Webhooks (eventos repo.content) | https://huggingface.co/docs/hub/webhooks | Doc oficial | 1 |
| 24 | User access tokens fine-grained | https://huggingface.co/docs/hub/security-tokens | Doc oficial | 1 |
| 25 | Run and manage Jobs (HF) | https://huggingface.co/docs/huggingface_hub/guides/jobs | Doc oficial | 1 |
| 26 | Vercel Functions Limits (timeouts) | https://vercel.com/docs/functions/limitations | Doc oficial | 1 |
| 27 | Edge Runtime streaming (SSE 25s/300s) | https://vercel.com/docs/functions/runtimes/edge | Doc oficial | 1 |
| 28 | Building Multi-Cloud GenAI Platforms (SkyPilot, R. Bhardwaj) | https://youtu.be/_oWJ0t-RW7o | Video | 1 |
| 29 | SkyPilot: Run AI on Any Cloud (Anyscale) | https://youtu.be/rgQxO54hN8Q | Video | 1 |
| 30 | Sky Computing keynote (Ion Stoica, USENIX) | https://youtu.be/AuNfxVLdo0A | Video | 1 |
