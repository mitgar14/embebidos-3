---
tipo: handoff
titulo: Rehacer entrenamiento minimo (deteccion de objetos) con destino HF Hub
fecha: 2026-05-26
hora: 00:44
rama: main
commit: 2f6f44d docs(06-03): summary, STATE, ROADMAP y REQUIREMENTS tras completar el codigo del plan
status: active
---

# Handoff: Rehacer entrenamiento minimo (deteccion de objetos) con destino HF Hub

Objetivo de la sesion (una linea): el usuario pidio TIRAR la maquinaria compleja de cloud training y rehacer el entrenamiento desde lo mas basico: entrenar YOLOv8n de 4 clases con el dataset que ya esta en HF, y dejar el modelo (best.pt + best.onnx) en HF Hub. Nada mas.

## 1. Estado actual

No hay modelo entrenado todavia. En HF `mitgar14/embebidos-3-models-v1d` solo existe `runs/heartbeat.jsonl` (obsoleto: ultima linea epoch 60 de una instancia ya destruida); `best.pt`, `best.onnx`, `manifests/manifest.json` y `manifests/eval_summary.json` dan 404. No hay ninguna instancia cloud corriendo (la ultima, una RTX 3060 en Vast, fue destruida en esta sesion porque segfalleaba; costo actual 0). La sesion previa se gasto entera peleando con la orquestacion Vast.ai (SkyPilot, onstart.sh, bootstrap.sh, watchdog, CommitScheduler) y nunca logro un run completo. El usuario corto eso de raiz: la nueva sesion arranca limpia y elige el camino MAS SIMPLE para entrenar y subir, sin resucitar nada de la maquinaria Vast salvo que se decida conscientemente.

## 2. Trabajo completado (en esta sesion, en su mayoria a DESCARTAR)

Lo de esta sesion fue diagnostico y andamiaje de Vast que el usuario decidio abandonar. Quedo en el arbol pero no es trabajo a continuar:

- `scripts/training/vast-run.sh`: lanzador Vast directo con 9 modos (sync, offers, launch, cheapest, ensure, fixtrain, gpucheck, wait, peek, hf). Funciona pero es el andamiaje complejo que se abandona. Util solo como referencia de comandos Vast (ver seccion 4).
- `scripts/training/onstart.sh:23-25`: arreglado el bug de `/workspace` inexistente (`mkdir -p` antes del `exec > >(tee ...)`).
- `scripts/training/bootstrap.sh:70-72`: torch cambiado a cu128 (`torch==2.7.0 torchvision==0.22.0`, index `cu128`).
- Instancia Vast 37865834 destruida (`vastai destroy instance 37865834 -y`).
- Panel web de monitoreo creado en sesiones anteriores (sigue valido como UI): `web/src/routes/entrenamiento.tsx`, `web/src/stores/trainingStore.ts`, `web/api/training-status.ts`.

## 3. Decisiones tomadas y por que

- **torch cu128 (2.7.0), no cu124 (2.4.1).** Sea cual sea la GPU cloud, cu128 corre en Ampere (RTX 30xx), Ada (40xx), Hopper y Blackwell (sm_120). cu124 revienta con "no kernel image available" en Blackwell. Si la nueva sesion entrena en cualquier GPU cloud reciente, instalar torch cu128. (En Colab/Kaggle el torch ya viene preinstalado y compatible: no tocarlo.)
- **El modelo va a HF Hub** repo `mitgar14/embebidos-3-models-v1d`. Es la persistencia acordada y lo que lee el panel web. No cambiar el repo destino.
- **Dataset y clases: v1d, 4 clases.** Orden fijo: `glass`(0), `paper`(1), `plastic`(2), `cardboard`(3). cardboard es la clase agregada (indice 3). El `data.yaml` del dataset en HF ya refleja esto; respetar ese orden o el modelo quedara desalineado con el frontend.
- **Destruir la RTX 3060 en vez de seguir parchandola.** Segfalleo dos veces en el backward (ver seccion 4); seguir gastando en una instancia flaky con el usuario pidiendo simplificar no tenia sentido. El usuario es fuertemente sensible al costo (historial: destruyo el mismo una Blackwell por cara).

## 4. Que se intento y NO funciono (NO reabrir estos caminos)

1. **SkyPilot sobre Vast.** Su catalogo cacheado anuncia "RTX_3090 a $0.25/h" pero Vast aprovisiona otra GPU (Blackwell RTX PRO 6000) a $1.478/h, y SkyPilot no tiene flag de tope de precio para Vast on-demand. Provisiono hardware/precio impredecible 3 veces. Abandonado (`train-v1d.sky.yaml`, `scripts/training/sky-run.sh` quedan como lapidas). NO reintentar SkyPilot+Vast.
2. **El segfault NO es del DataLoader ni de `/dev/shm`.** El traceback dice "DataLoader worker killed by signal: Segmentation fault" pero el crash real esta en `torch.autograd.backward` (PyTorch lo reporta via su handler SIGCHLD). Probado: `/dev/shm` era de 10G (de sobra) y poner `workers=0` en `model.train()` NO lo arreglo (segfalleo igual, en el epoch 0). El unico run que llego a epoch 60 fue en la Blackwell cara, nunca en la RTX 3060 barata. Hipotesis no confirmada (no se corrio el `gpucheck`): GPU/driver flaky de esa instancia barata puntual. Leccion: las instancias Vast mas baratas pueden tener hardware no confiable; un segfault en el backward no se cura con `workers=0`.
3. **`vastai execute` solo corre en instancias DETENIDAS** ("Execute command only avail on stopped instances"). No sirve para inspeccionar una instancia viva.
4. **SSH a una instancia Vast ya arrancada falla** ("Permission denied (publickey)"): las claves se inyectan al boot. Para entrar en caliente: `vastai attach ssh <id> "<pubkey>"` inyecta la clave en la instancia corriendo (esto SI funciono). `vastai logs` solo muestra el stdout del contenedor (onstart), no la salida del notebook dentro de tmux.
5. **Doble borde de shell Bash-tool -> MINGW -> wsl.exe -> Ubuntu destroza comandos:** mete CRLF en variables y rompe `for` inline y `$()` anidados dentro de `echo "..."` (sintoma tipico: etiquetas vacias y curl devolviendo `000`). Lo unico robusto fue: escribir scripts `.sh` con finales LF y heredocs `<<'EOF'` de comillas SIMPLES, ejecutarlos con `wsl.exe -d Ubuntu bash -lc 'bash /mnt/c/.../script.sh'` y limpiar el output con `tr -d '\0' | tr -d '\r'`. Si la nueva sesion usa cloud-via-CLI desde esta maquina, respetar ese patron.
6. **El auto-destroy del notebook y el cron watchdog usan `vastai destroy` SIN `-y`** (`bootstrap.sh:149`), asi que se quedan esperando confirmacion y la instancia NUNCA se autodestruye. Si se vuelve a usar Vast, destruir a mano con `-y` o arreglar esos call sites.

## 5. Proximos pasos

1. **Elegir la plataforma de training mas simple.** El job es minusculo (163 imagenes, YOLOv8n, ~100 epochs = pocos minutos de GPU), entra de sobra en cualquier free tier. Candidatas, de mas a menos simple: Google Colab (GPU T4 gratis, notebook directo, cero orquestacion) o Kaggle (30 h/semana de GPU; hay MCP de Kaggle disponible en el entorno). Evitar Vast/SkyPilot salvo que se necesite una GPU mas grande, que aca no hace falta. Esta es una decision de arranque: tomarla primero.
2. **Escribir un notebook MINIMO** (partir de `notebooks/train_v1d_vastai.ipynb` celda 20, pero QUITANDO callbacks, heartbeat, emergency-export y CommitScheduler). El flujo basico es: (a) `pip install ultralytics huggingface_hub`; (b) bajar el dataset con `huggingface_hub.snapshot_download(repo_id="mitgar14/embebidos3-dataset-v1d", repo_type="dataset", ...)` usando HF_TOKEN; (c) `YOLO("yolov8n.pt").train(data="<ruta>/data.yaml", epochs=100, imgsz=640, workers=2)`; (d) `model.export(format="onnx", opset=11)`; (e) subir `best.pt` y `best.onnx` a `mitgar14/embebidos-3-models-v1d` con `HfApi.upload_file`. El augmentation reforzado del notebook viejo (copy_paste=0.3, mixup=0.15, de la investigacion 2026-05-15 sobre desbalanceo) es opcional: conservarlo si se quiere, pero NO es necesario para el MVP.
3. **Verificar artefactos en HF y decidir el contrato con el panel web.** El panel (`web/api/training-status.ts`) espera en HF: `runs/heartbeat.jsonl`, `manifests/eval_summary.json`, `manifests/manifest.json`. Un training basico no genera heartbeat en vivo, asi que el panel quedara en "waiting/offline". Decidir conscientemente: o el notebook minimo escribe al menos `manifests/eval_summary.json` (metricas finales) y `manifests/manifest.json` para que el panel muestre el resultado final, o se simplifica el panel para que solo refleje "modelo disponible". El minimo util para el panel es `eval_summary.json` con mAP50/mAP50_95/precision/recall + per_class.

## 6. Gotchas y minas

- **HF_TOKEN** vive en `.env` del repo (raiz). NO copiarlo a ningun lado versionado. La nueva sesion lo lee de ahi.
- **uv/uvx y la CLI de Vast viven en WSL (Ubuntu), no en Windows nativo.** `python` de PowerShell sirve para scripts locales sueltos (ej. el de handoff), pero el stack de training (uv, vastai) corre en WSL.
- **Orden de clases es un contrato, no un detalle.** glass/paper/plastic/cardboard = 0/1/2/3. El frontend mapea por indice (`CLASS_LABELS` en `web/src/routes/entrenamiento.tsx`). Si el `data.yaml` reordena, el panel etiqueta mal.
- **El panel web usa formato es-CO** (coma decimal). No es bug.
- **No quedo nada corriendo en cloud.** Antes de asumir que hay GPU, la nueva sesion debe provisionar/abrir su entorno elegido.

## 7. Archivos clave

- `notebooks/train_v1d_vastai.ipynb`: celda 20 = `model.train(...)` con CFG (epochs, imgsz, batch, augmentation) + callbacks. Es la FUENTE de la que extraer la version minima. Todo lo de heartbeat/CommitScheduler/emergency-export es lo que se quita.
- `scripts/training/vast-run.sh`: andamiaje Vast que se abandona. Util solo como recetario de comandos Vast (`search offers`, `create instance`, `attach ssh`, `destroy -y`) si se decidiera volver a Vast.
- `web/api/training-status.ts`, `web/src/stores/trainingStore.ts`, `web/src/routes/entrenamiento.tsx`: el panel de monitoreo. Define el contrato de artefactos que el training deberia dejar en HF (ver paso 3).
- `train-v1d.sky.yaml`, `scripts/training/sky-run.sh`, `scripts/training/onstart.sh`, `scripts/training/bootstrap.sh`: maquinaria abandonada (SkyPilot/Vast). No continuar sobre ellos.

## 8. Artefactos durables (referencia, no copiar)

- Dataset (4 clases v1d): https://huggingface.co/datasets/mitgar14/embebidos3-dataset-v1d
- Modelo destino: https://huggingface.co/mitgar14/embebidos-3-models-v1d
- Repo raw-batches/.notebook: https://huggingface.co/datasets/mitgar14/embebidos3-raw-batches
- Investigacion de la orquestacion Vast (por que fallo): `investigaciones/2026-05-25-automatizacion-vastai-training-orquestacion.md`
- Decisiones de producto/frontend y naming (Tiny Trash): `MEMORY.md` del proyecto y los `memory/feedback_*.md`.
- Commit base: `2f6f44d`.

## 9. Estado del entorno

- Cloud: SIN instancias corriendo (Vast 37865834 destruida). Costo 0.
- HF `mitgar14/embebidos-3-models-v1d`: solo `runs/heartbeat.jsonl` obsoleto; sin modelo ni metricas (best.pt/best.onnx/manifest.json/eval_summary.json = 404).
- git: rama `main`, con cambios sin commitear de esta sesion (vast-run.sh con modos nuevos, onstart.sh, bootstrap.sh, archivos de planning/web). Decidir si commitear el andamiaje o descartarlo antes de empezar limpio.
- Secretos: HF_TOKEN en `.env`; VAST_API_KEY en `~/.config/vastai/vast_api_key` (WSL). No incluidos aqui.

## 10. Archivos modificados (auto)

- `.planning/REQUIREMENTS.md`
- `.planning/ROADMAP.md`
- `.planning/STATE.md`
- `scripts/training/bootstrap.sh`
- `scripts/training/onstart.sh`
- `web/bun.lock`
- `web/package.json`
- `web/src/components/Placeholder.tsx`
- `web/src/index.tsx`
- `web/src/lib/detection.ts`
- `web/src/lib/mqtt.ts`
- `web/src/lib/servoProtocol.ts`
- `web/src/routes/control.tsx`
- `web/src/routes/labelling.tsx`
- `web/src/stores/servoStore.ts`
- `investigaciones/2026-05-25-automatizacion-vastai-training-orquestacion.md`
- `investigaciones/2026-05-25-camara-local-server-jetson.md`
- `models/`
- `notebooks/train_v1d_vastai.ipynb`
- `scripts/training/sky-run.sh`
- `scripts/training/vast-run.sh`
- `train-v1d.sky.yaml`
- `web/api/`
- `web/src/routes/entrenamiento.tsx`
- `web/src/stores/trainingStore.ts`

## 11. Prompt de reanudacion

> Lee `.claude/handoffs/2026-05-26-training-basico-hfhub.md` y continua desde ahi. Objetivo: rehacer el entrenamiento desde lo MAS BASICO, entrenar YOLOv8n de 4 clases (glass/paper/plastic/cardboard) con el dataset HF `mitgar14/embebidos3-dataset-v1d` y subir best.pt + best.onnx a `mitgar14/embebidos-3-models-v1d`. Empieza por el paso 1 de "Proximos pasos": elegir la plataforma mas simple (Colab o Kaggle, free GPU; el job es minusculo). RESTRICCION CRITICA: NO reabrir SkyPilot ni la orquestacion Vast (onstart/bootstrap/watchdog/CommitScheduler); el detalle de por que fallaron esta en "Que se intento y NO funciono". Respeta el orden de clases 0=glass,1=paper,2=plastic,3=cardboard.
