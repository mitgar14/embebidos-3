# 03 · HuggingFace Hub integration

Cliente HTTP minimalista para operar contra `mitgar14/embebidos-3-models` sin instalar `huggingface_hub` (incompatible con Py3.6 del Nano y pesado para un MVP).

## Archivos

| Archivo | Propósito |
|---|---|
| `scripts/hf_rest.py` | Cliente REST único — `urllib` puro |

## Por qué REST manual y no la lib oficial

- `huggingface_hub` requiere Py3.7+; Nano corre Py3.6
- Dependencias pesadas para un caso de uso de 2 endpoints
- Necesitamos validación SHA256 de archivos LFS manualmente — la lib la hace pero opaco

## Operaciones que soporta

```bash
# Descargar archivo (resolviendo LFS pointer si aplica)
python3 hf_rest.py download <path-in-repo> <local-path> [--revision <sha>]

# Obtener la HEAD actual del repo
python3 hf_rest.py head

# Obtener metadata del último commit (autor, fecha, mensaje)
python3 hf_rest.py commit-info [--revision <sha>]

# Resolver SHA256 de un archivo LFS sin descargarlo
python3 hf_rest.py lfs-sha <path-in-repo> [--revision <sha>]
```

## Autenticación

Lee `HF_TOKEN` de env. Para los endpoints que tocan HF públicos no es estrictamente necesario, pero el repo está en private — sí lo necesita. Se inyecta vía `EnvironmentFile=/etc/embebidos3/secrets.env` del unit del builder.

El server FastAPI también lee `HF_TOKEN` desde env para el endpoint `/model/check-updates`. Se carga en `nano_start_server.sh` con `set -a; source /etc/embebidos3/secrets.env; set +a`.

## Estructura del repo HF

```
mitgar14/embebidos-3-models/
├── manifests/
│   └── manifest.json       # SSOT: artifacts + recovery info
└── exports/
    └── best.onnx           # LFS-tracked, el modelo
```

El manifest tiene shape:

```json
{
  "version": "1",
  "hf_revision": "b93964f9e4f9464cfe55b13ca5a577ba383a4dd5",
  "hf_commit_date": "2026-05-16T14:47:01.000Z",
  "artifacts": {
    "best_onnx": {
      "path": "exports/best.onnx",
      "sha256": "223f1a71c4b1bd08effdfa02fabb1ce259a3a507015d39e2359b5bec3dc805ad",
      "size_bytes": 11567823
    }
  },
  "training": {
    "dataset_revision": "v3",
    "epochs": 240,
    "imgsz": 416,
    "classes": ["glass", "paper", "plastic"]
  },
  "recovery": {
    "hf_revision": "b93964f..."   // mismo que arriba; redundancia para auditar manifest huérfano
  }
}
```

## Resolución de LFS pointer

Cuando HF devuelve un archivo LFS, la primera respuesta es un *pointer file* tipo:
```
version https://git-lfs.github.com/spec/v1
oid sha256:223f1a71...
size 11567823
```

El cliente `hf_rest.py` detecta esto, extrae el OID y hace un segundo request al endpoint LFS-batch para obtener la URL real (S3 firmado). Luego baja el binario y verifica SHA256.

Esta es la razón por la que `lfs-sha` puede consultar el SHA sin descargar: el SHA está en el pointer mismo.

## Endpoint `/model/check-updates` (server)

Llama a `hf_rest.py` para:
1. Obtener HEAD actual de HF Hub
2. Obtener SHA256 del ONNX del HEAD
3. Comparar con `current_revision` y `current_onnx_sha256` del meta local

Devuelve:
```json
{
  "ok": true,
  "current_revision": "b93964f...",
  "latest_revision": "c0a681...",
  "current_onnx_sha256": "223f1a71...",
  "latest_onnx_sha256": "223f1a71...",
  "same_revision": false,
  "same_onnx": true,
  "up_to_date": true,
  "has_engine": true
}
```

### Por qué comparar tanto revision como onnx_sha

Un commit en HF puede ser cosmético (README, manifest update sin cambiar el modelo). El SHA del ONNX **siempre** refleja si el modelo cambió. La combinación da 4 casos útiles que el dashboard usa para mostrar el mensaje correcto:

1. **`up_to_date && same_revision`** → "Modelo al día"
2. **`up_to_date && !same_revision`** → "Modelo al día (commit cosmético)" — mismo modelo, nuevos commits sin tocar pesos
3. **`!up_to_date && !same_revision`** → "Nueva iteración disponible" — recompilar
4. **`!same_onnx && same_revision`** → "Inconsistencia" — caso patológico (alguien sobreescribió el ONNX en el mismo commit)

Implementado en `scripts/dashboard/modelo.js::formatCheckUpdates()`.

## Endpoint `/model/build` flow respecto a HF

El builder (ver doc 04) hace los siguientes pasos relacionados a HF:
1. Descarga `manifests/manifest.json`
2. Parsea `recovery.hf_revision` (o fallback `hf_revision` o `"main"`)
3. Descarga `exports/best.onnx` **con esa revision específica** (no `main`) — garantiza idempotencia
4. Verifica SHA256 contra `manifest.artifacts.best_onnx.sha256`
5. Si difiere → exit 2 con `[BUILD] SHA mismatch`

## Gotchas

- **HF rate limiting**: el `check-updates` puede hacer hasta 3 requests (HEAD + LFS resolve + commit-info). Cachear no aplica porque el usuario está esperando el resultado.
- **HF puede tardar**: en caso de Nano con red Tailscale lenta, `hf_rest.py download` para un ONNX de 11 MB puede tardar 5-15s. El endpoint `/model/check-updates` solo hace HEAD requests, mucho más rápido.
- **El token expira**: si se rota, hay que actualizar `/etc/embebidos3/secrets.env` y `sudo systemctl restart embebidos3-server.service`.
