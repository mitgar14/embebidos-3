# HF Hub upload de binarios desde Python 3.6 sin `huggingface_hub` (MVP)

Dominio: arquitectura del backup/versionado del engine TensorRT del proyecto **embebidos-3** (Jetson Nano B01, JetPack 4.6.1, Python 3.6.9, TensorRT 8.2.1.8, Maxwell sm_53). El binario a respaldar es un engine TRT FP16 de ~13,5 MB compilado en el propio Nano.

Origen de la investigación: bug **V-1** del pipeline dashboard, documentado en `agent-docs/VACIOS.md` y `agent-docs/HANDOFF-2026-05-16-fix-v1-backup-hf.md`. El cliente REST minimal `scripts/hf_rest.py` envía un payload mal formado al endpoint `POST /api/models/<repo>/commit/main`, HF responde **400 Bad Request**, el builder aborta con `exit 4` y nunca promueve el engine recién compilado.

---

## Ronda 1 — 2026-05-16 16:00:00 (medio)

### Pregunta de partida

¿Cuál de las tres opciones de fix planteadas en el handoff (A backup local puro · A+ híbrida con manifest en HF · B fix completo con flujo LFS) es la más sustentada por la comunidad y el código fuente actual de `huggingface_hub`, considerando que el cliente corre en Python 3.6 sobre Jetson Nano sin posibilidad de instalar la lib oficial?

### Track A — agentes de research

#### A.1 · `research-code` — source code de `huggingface/huggingface_hub`

**Shape exacto del endpoint commit (confirmado en `_commit_api.py:876-960` y `hf_api.py:5019-5038`):**

```
POST https://huggingface.co/api/models/<namespace>/<repo>/commit/<branch>
Content-Type: application/x-ndjson
Authorization: Bearer <token>
```

El body **no** es JSON convencional. Es **NDJSON** (un objeto JSON por línea). La primera línea siempre es `header`. Cada línea tiene exactamente dos claves: `"key"` y `"value"`.

| `key` | Campos de `value` | Notas |
|---|---|---|
| `header` | `summary`, `description`, `parentCommit?` | Obligatorio, siempre primero |
| `file` | `path`, `content` (base64), `encoding: "base64"` | Solo regulares ≤ 10 MB |
| `lfsFile` | `path`, `algo: "sha256"`, `oid`, `size` | Para LFS |
| `deletedFile` | `path` | Eliminar archivo |
| `deletedFolder` | `path` | Eliminar carpeta |

Ejemplo NDJSON completo para un binario LFS de 13 MB:

```ndjson
{"key":"header","value":{"summary":"Add engine","description":"","parentCommit":"abc..."}}
{"key":"lfsFile","value":{"path":"engines-archive/.../best_fp16.engine","algo":"sha256","oid":"e3b0...","size":13631488}}
```

Respuesta exitosa:

```json
{"commitUrl":"https://huggingface.co/<ns>/<repo>/commit/<sha>","commitOid":"<sha>"}
```

**Por qué falla `hf_rest.py` actual:** el shape `{"summary":..., "files":[{"path":..., "encoding":"base64", "content":"<b64>"}]}` **no existe en ninguna parte de la API**. Es invento del cliente actual. Aunque el shape fuera correcto, para 13,5 MB el server respondería `uploadMode: "lfs"` y no aceptaría contenido inline.

**Preupload (`_commit_api.py:700-780`):**

```
POST /api/models/<repo>/preupload/<branch>
Content-Type: application/json
```
```json
{"files":[{"path":"engine","sample":"<b64 primeros 512 bytes>","size":13631488}],"gitIgnore":"*.log"}
```

Respuesta por archivo:

```json
{"files":[{"path":"engine","uploadMode":"lfs","shouldIgnore":false,"oid":null}]}
```

El threshold de inline-vs-LFS es **~10 MB** y se cruza con `.gitattributes` del repo. La muestra de 512 bytes le permite al server detectar binarios independientemente del tamaño. Archivos de tamaño 0 se fuerzan a `regular`.

**LFS batch + PUT S3 + Verify (`lfs.py:100-265`):**

1. `POST <repo>.git/info/lfs/objects/batch` con `Content-Type: application/vnd.git-lfs+json`:
   ```json
   {"operation":"upload","transfers":["basic","multipart"],
    "objects":[{"oid":"e3b0...","size":13631488}],
    "hash_algo":"sha256","ref":{"name":"main"}}
   ```
   Respuesta para archivo nuevo: `actions.upload.href` (URL S3 firmada) + opcional `actions.verify.href`. Para 13 MB con `chunk_size` típico de 5 MB → 3 PUTs por partes + POST de completion con los ETags recolectados.

2. `PUT <presigned_url>` con el binario como stream (sin Authorization — la URL ya está firmada).

3. Si el server devuelve `verify.href`: `POST <verify> {"oid":"e3b0...","size":13631488}` (opcional, solo si el server lo pidió).

**Estimación realista de LOC para reimplementar el flujo completo en Py3.6 + `requests`:**

| Módulo | LOC funcional |
|---|---|
| `sha.py` (hash + sample 512B) | 20-30 |
| `preupload.py` | 40-60 |
| `lfs_batch.py` | 50-70 |
| `lfs_upload.py` (PUT básico + multipart + verify) | 80-120 |
| `commit.py` (NDJSON builder + POST) | 50-70 |
| `error_handling.py` | 30-50 |
| `main.py` (orquestador) | 40-60 |
| **Total funcional sin tests** | **310-460 LOC** |
| Con tests + manejo robusto | 600-800 LOC |

**Tiempo realista: 1-2 días de trabajo** (no 2-4 horas como sugería el handoff). Mayor riesgo: la clase `SliceFileObj` (`lfs.py:370-395`) que maneja el multipart sin el `io` avanzado de Py3.7+.

**Última versión de `huggingface_hub` que soporta Py3.6:** v0.4.0 (`python_requires=">=3.6.0"`). v0.5.0 ya pide Py3.7. **Pero**: el protocolo NDJSON se introdujo en PR [#1117](https://github.com/huggingface/huggingface_hub/pull/1117) (mergeada 2022-10-21), correspondiente a versiones ≥ 0.12.x. Por tanto **no existe ninguna versión de la lib que combine Py3.6 + NDJSON**. La única vía desde Py3.6 es reimplementar a mano.

**Implementación de referencia open-source minimal en Py3.6:** búsqueda exhaustiva (gh search repos, gh search code, Exa) — no existe ninguna. Solo el source oficial sirve de referencia.

#### A.2 · `research-web` — comunidad y alternativas

**Estado Py3.6 + HF Hub en la comunidad:** Py3.6 EOL desde diciembre de 2021. Issue [#786](https://github.com/huggingface/huggingface_hub/issues/786) ya planteaba el drop en 2022. **No hay workarounds publicados** por la comunidad para este caso — es nicho (hardware legado como Jetson Nano con JetPack fijo).

**Insight central del agente:** *el upload del engine no tiene por qué ocurrir desde el Nano*. El entrenamiento corre en Vast.ai (con Python moderno), el laptop tiene SSH al Nano. El upload puede delegarse fuera del device, eliminando toda la complejidad de implementar LFS sobre Py3.6 sin lib oficial.

**Top 5 alternativas para hostear el binario de ~13 MB:**

| Opción | Score MVP | Pros | Contras |
|---|---|---|---|
| **GitHub Releases** | **9/10** | Trivial `requests.get`; sin límite storage; sin límite bandwidth; sin creds en el Nano (repo público); durable como Microsoft | Sin UI de modelo; upload desde laptop/CI, no Nano |
| HF Hub REST directa | 7/10 | URL estable; tarjeta de modelo; mismo repo que el ONNX | Flujo 3 pasos complejo; storage LFS no se libera al borrar (GC eventual); private es paid |
| Cloudflare R2 | 6/10 | Egress gratis; 10 GB free | Requiere tarjeta de crédito; setup más complejo; sin UI ML |
| Backblaze B2 | 6/10 | 11 nines durabilidad; egress 3x storage/mes free | Requiere tarjeta; setup similar a R2 |
| git-lfs CLI subprocess | 3/10 | Técnicamente compatible Py3.6 | Stateful (clone en SD del Nano); setup git-lfs en Ubuntu 18.04 manual; working tree puede quedar inconsistente |
| HF Storage Buckets | 5/10 | Xet dedup; CDN | CLI requiere hub moderno (no Py3.6); experimental |

**Patrón "manifest separado del binario"**: encontrado en proyectos académicos y de producción. El manifest JSON liviano vive en repo Git (versionado), el binario vive en object storage. Ejemplos citados:

- **OmniBioAI Model Registry** ([github.com/man4ish/omnibioai-model-registry](https://github.com/man4ish/omnibioai-model-registry), 2026) — manifest en SQLite local, binarios en S3, CLI + FastAPI.
- **TinyMLDelta** ([medium 2025](https://medium.com/@felixgalindo91/introducing-tinymldelta-incremental-ml-model-updates-for-tiny-devices-96663edd1991)) — OTA para TinyML embedded, manifest decide si aplica patch incremental.
- **DVC** ([dvc.org](https://dvc.org)) — archivos `.dvc` en git (hash + remote URL), binarios en S3/GCS/Azure.
- **MLflow** — model registry separa metadata de artifact storage.

Plantilla de manifest recomendada por el agente (adaptable a embebidos-3):

```json
{
  "schema_version": "1.0",
  "model_id": "yolov8n-trt-fp16-sm53",
  "version": "2026-05-16T18:30:00Z",
  "hardware": {"platform":"jetson-nano-b01","jetpack":"4.6.1","tensorrt":"8.2.1.8","arch":"sm_53","precision":"fp16"},
  "artifact": {
    "filename":"best_fp16.engine",
    "size_bytes":13566282,
    "sha256":"<sha>",
    "download_url":"https://github.com/<user>/<repo>/releases/download/<tag>/best_fp16.engine"
  },
  "training": {"source_onnx_sha256":"<sha>","imgsz":416,"classes":["glass","paper","plastic"]}
}
```

**Veredicto sobre git-lfs vía subprocess:** técnicamente factible pero no recomendado. Requiere clonar el repo HF en el Nano (estado persistente en SD), instalar git-lfs en Ubuntu 18.04 manualmente, hacer `git pull` antes de cada push, y manejar working trees inconsistentes si la red falla a mitad. Para MVP académico, el overhead operativo no compensa.

**Costos / cuotas relevantes (confirmados):**

- **GitHub Releases:** sin límite de storage para release assets, sin límite de bandwidth para downloads, hasta 2 GB por asset. No expiran. ([docs.github.com](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases), discussion [73875](https://github.com/orgs/community/discussions/73875)).
- **HF Hub público:** "best-effort" sin cuota numérica fija para repos públicos.
- **HF Hub privado:** 100 GB en free tier.
- **Storage de LFS borrado:** GC eventual no garantizado (confirmado por Wauplin en foro [146721](https://discuss.huggingface.co/t/will-lfs-related-functionality-come-to-hf-api/146721)). Para builds frecuentes, el storage crece monotónicamente.

**Tendencia futura del Hub:** desde v0.30.0 (2024) HF migra de LFS a **Xet** (dedup a nivel chunk de 64 KB). 77 PB+ ya migrados. LFS sigue funcionando pero la inversión está en Xet (blog [`huggingface-hub-v1`](https://huggingface.co/blog/huggingface-hub-v1)).

### Track B — búsqueda ampliada

#### Fase 1 — `discover.py` (Exa Search)

21 fuentes descubiertas. Selección priorizada para Fase 2 (no duplicar Track A, foco en evidencia primaria del bug):

- Foro HF · [HF Hub Commit API isn't accepting LFS files](https://discuss.huggingface.co/t/hf-hub-commit-api-isnt-accepting-lfs-files/113997) — **caso espejo** de alguien reimplementando exactamente nuestro flujo.
- Foro HF · [Will LFS related functionality come to hf_api?](https://discuss.huggingface.co/t/will-lfs-related-functionality-come-to-hf-api/146721) — intervención de Wauplin (maintainer) sobre LFS y storage.
- Docs oficiales · [Upload files to the Hub v1.7.1](https://huggingface.co/docs/huggingface_hub/v1.7.1/en/guides/upload).
- Blog HF · [huggingface_hub v1.0: Five Years](https://huggingface.co/blog/huggingface-hub-v1) — historia API HTTP, migración a Xet.
- Hexdocs · [HfHub.Commit Elixir](https://hexdocs.pm/hf_hub/HfHub.Commit.html) — **referencia minimal de reimplementación en otro lenguaje**, confirma threshold 10 MB.
- TheNeuralBase · [Model upload and versioning](https://theneuralbase.com/huggingface-api/learn/advanced/model-upload-and-versioning/).

GitHub source confirmado para deep-dive del Track A: `_commit_api.py`, `lfs.py`, `_upload_large_folder.py`, `cli/upload.py`, `hf_api.py`.

PRs históricas relevantes: [#888](https://github.com/huggingface/huggingface_hub/pull/888) (intro `create_commit`), [#1117](https://github.com/huggingface/huggingface_hub/pull/1117) (NDJSON), [#1699](https://github.com/huggingface/huggingface_hub/pull/1699) (preupload LFS antes de commit), [#920](https://github.com/huggingface/huggingface_hub/pull/920) (chunking del preupload).

#### Fase 1b — MCP `youtube`

`youtube_search` con query combinada `huggingface hub upload large files python LFS commit api | huggingface_hub commit api architecture`, `top_k=8`, `include_long_tail=true` (200u quota).

**Resultado: 0 hits.** Gap registrado. El tema es demasiado nicho técnico — no hay charlas ni tutoriales en video sobre reimplementación del flujo commit/LFS de HF sin la lib oficial. Quota usada: 200u (de los 10.000u/día). No se justificó bajar `aai_threshold` ni explorar AAI fallback porque no había videos candidato.

#### Fase 2 — Exa `crawling_exa` sobre URLs prioritarias

**Foro [113997](https://discuss.huggingface.co/t/hf-hub-commit-api-isnt-accepting-lfs-files/113997) (caso espejo):** alguien reimplementando el commit API en Rust llegó al mismo punto que nosotros (400 al hacer commit con LFS). Logró preupload + S3 PUT + completion (`{"success":true}`) pero el `/commit/main` rechazaba con `"Your push was rejected because an LFS pointer pointed to a file that does not exist"`. La intervención de **celinah** (HF staff) reveló un trap no documentado oficialmente:

> El SHA256 enviado al verify y el `total_bytes` deben incluir los primeros 1024 bytes (`sample`) del archivo. Si el sample se hashea aparte y `total_bytes` no lo cuenta, la size enviada es off-by-1024 y el commit falla aunque el binario esté correctamente subido a S3.

El payload del commit en ese hilo confirma exactamente el shape NDJSON descrito por research-code:

```ndjson
{"key":"header","value":{"summary":"Upload 20m_file.bin with hf_hub","description":"lalalala"}}
{"key":"lfsFile","value":{"path":"20m_file.bin","algo":"sha256","oid":"7d34cce2c40a7089ea8b1d8ea9c25c573c46fcce7aa60579748118183a03f272","size":20970496}}
```

**Foro [146721](https://discuss.huggingface.co/t/will-lfs-related-functionality-come-to-hf-api/146721) (Wauplin, storage management):**

- `super_squash_history` no libera storage S3 inmediatamente — GC eventual, no garantizado.
- Private storage es ahora servicio pagado.
- Para casos de muchos checkpoints temporales, Wauplin recomienda `delete_repo` + `create_repo` (perdés historial pero liberás storage).
- PR [#2954](https://github.com/huggingface/huggingface_hub/pull/2954) agrega `permanently_delete_lfs_files` pero requiere la lib moderna.

**Hexdocs [HfHub.Commit Elixir](https://hexdocs.pm/hf_hub/HfHub.Commit.html) (referencia minimal de reimplementación independiente):**

> "Files are uploaded in one of two modes: **Regular: Base64-encoded in commit payload (for files < 10MB)** — **LFS: Git Large File Storage protocol (for files >= 10MB)**. The upload mode is automatically determined based on file size."

`lfs_threshold()` retorna 10 MB exactos. Esta lib en Elixir es la única reimplementación pública minimal del protocolo y **confirma el threshold de 10 MB con evidencia de tercero**, independiente del source oficial Python.

**Blog [`huggingface-hub-v1`](https://huggingface.co/blog/huggingface-hub-v1) (contexto histórico):**

- La HTTP Commit API se introdujo en v0.8.1 (junio 2022). Antes era Git wrapper.
- v0.30.0 (2024) introdujo Xet (chunk-level dedup, 64 KB chunks).
- v1.0 (octubre 2025) removió `Repository` class git-based, migró de `requests` a `httpx`, `hf_xet` reemplazó `hf_transfer`.
- 113.5 millones de downloads/mes. 200k+ repos dependen. Ecosistema masivo.

**Docs oficiales v1.7.1** (`huggingface.co/docs/huggingface_hub/v1.7.1/en/guides/upload`): describen exclusivamente la API high-level (`upload_file`, `upload_folder`, `create_commit`). El protocolo HTTP subyacente **no está documentado en docs oficiales** — solo en source code. Esto explica por qué el cliente actual inventó un shape: no hay spec pública del NDJSON.

### Síntesis: confrontación de las opciones

| Criterio | A pura (local) | A+ híbrida (local + manifest HF) | A+ con GitHub Releases | B (LFS completo) |
|---|---|---|---|---|
| **LOC nuevas/modificadas** | ~30 (modificar paso 9 del builder) | ~80 (lo de A + arreglar shape NDJSON commit.py para subir el manifest <10MB inline) | ~30 (igual que A; el upload a Releases lo hace otra máquina) | 310-460 |
| **Tiempo realista** | 30 min impl + 30 min validar | 90 min impl + 30 min validar | 30 min impl + 30 min validar Nano; ~1 h script complementario en laptop (opcional, no esta sesión) | 1-2 días + tests |
| **Robustez del backup remoto** | Ninguna (solo SD del Nano) | Catálogo remoto en HF (manifest), binario solo local | Binario+manifest en GitHub Releases (durable, sin LFS) | Binario completo en HF (durable, con LFS) |
| **Riesgo de bug nuevo** | Bajo | Bajo (un solo endpoint a arreglar, shape público confirmado) | Bajísimo | Alto (multipart S3, sample-en-hash, sin referencia open-source en Py3.6) |
| **Tendencia futura** | Neutra | Neutra (manifest es inmune a LFS→Xet) | Inmune al Hub | Acoplado a LFS, que HF está deprecando lentamente a favor de Xet |
| **Storage growth** | Lineal en SD (~13 MB/build) | Igual A en Nano + ~1 KB/build en HF | Igual A en Nano + 13 MB/build en GitHub (sin límite) | Lineal en LFS HF; no liberable al borrar |
| **Creds escritura en el Nano** | No requiere | HF_TOKEN ya está | HF_TOKEN ya está (para manifest); GitHub no necesita creds en Nano | HF_TOKEN ya está |
| **Útil para sustentación MVP** | Sí | Sí | Sí | Overkill |
| **Coherencia con `agent-docs` y CLAUDE.md** | Aligned | Aligned | Aligned | "MVP académico" del CLAUDE.md desalienta |

### Decisión recomendada (sustento)

**Recomendación primaria: A+ híbrida en su variante "manifest en HF + binario local en Nano"**, con la **opción explícita** de extender más adelante a A+ con GitHub Releases si se quiere backup remoto durable del binario.

Razones cruzadas:

1. **El shape NDJSON está confirmado y es estable.** La PR [#1117](https://github.com/huggingface/huggingface_hub/pull/1117) lo introdujo en 2022 y el blog v1.0 (octubre 2025) confirma que la HTTP Commit API es la base sobre la que se construyen las APIs modernas. Arreglar el `upload_file_inline` para usar NDJSON + `application/x-ndjson` es un cambio quirúrgico de ~50 LOC en `commit.py` (parte del módulo `hf_rest.py`).

2. **Subir solo el manifest (<1 KB JSON) elimina toda la complejidad de LFS.** El manifest cae cómodamente por debajo del threshold de 10 MB, va con `"key":"file","value":{"encoding":"base64","content":"<b64>"}` inline, sin preupload, sin batch, sin S3 PUT, sin verify. El trap del sample-en-hash que documentó celinah no aplica.

3. **El binario local en `engines-archive/<ts>__<sha>/` en el Nano** mantiene la idempotencia del builder y permite rollback inmediato sin red, que es el caso de uso real durante la sustentación. La pérdida de la SD es aceptable para MVP académico (alineado con CLAUDE.md: *"busca construir un MVP. Las implementaciones que veas que son complicadas realmente están orientadas a ofrecer el funcionamiento más fluido durante la sustentación"*).

4. **El research-web encontró el patrón "manifest separado del binario"** en DVC, MLflow, OmniBioAI, TinyMLDelta. Es estándar de industria para devices embebidos.

5. **La opción B (LFS completo) tiene 3 problemas graves:**
   - Tiempo real 1-2 días vs estimación handoff 2-4 h.
   - Storage HF privado/LFS borrado **no se libera** (Wauplin confirma) — para builds frecuentes, crece monotónicamente.
   - HF está migrando a Xet; implementar LFS a mano queda acoplado a un protocolo en lento decline.
   - Sin referencia open-source minimal en Py3.6 — somos los primeros.

6. **GitHub Releases como follow-up (no esta sesión):** un script complementario que corra **desde el laptop** (no desde el Nano), pulle los `engines-archive/<ts>__<sha>/` del Nano via SSH y los suba a GitHub Releases con `gh release create`. Esto da backup remoto durable del binario sin tocar Py3.6 ni LFS, y aprovecha que GitHub Releases tiene bandwidth/storage sin límite explícito. Documentar como V-1.1 en `VACIOS.md`.

**Alternativa razonable: A pura.** Si la prioridad es máxima simplicidad para la sustentación y el catálogo remoto se considera valor agregado innecesario, A pura es defendible. Diferencia con A+ HF: pierde ~60 min de implementación y la searchability del manifest, gana cero riesgo de regresión por un nuevo endpoint.

**Descartar B.** No por imposibilidad técnica (es factible en 1-2 días), sino por desbalance costo/beneficio dado el contexto MVP académico, la tendencia LFS→Xet, y la ausencia de referencias open-source. Si en el futuro alguien quiere backup binario remoto durable, la ruta es GitHub Releases (5 min de `gh release create`) o S3-compatible (R2/B2 con boto3 desde laptop), no implementar LFS-HF a mano.

---

## Historial de investigación

| Ronda | Fecha | Profundidad | Foco |
|---|---|---|---|
| 1 | 2026-05-16 | medio | Bug V-1: shape commit API + viabilidad reimplementar LFS desde Py3.6 + alternativas comunidad |

---

## Fuentes consultadas (acumulado)

| # | Título | URL | Tipo | Ronda |
|---|---|---|---|---|
| 1 | HF Hub Commit API isn't accepting LFS files (foro) | https://discuss.huggingface.co/t/hf-hub-commit-api-isnt-accepting-lfs-files/113997 | Foro HF (caso espejo) | 1 |
| 2 | Will LFS related functionality come to hf_api? (foro) | https://discuss.huggingface.co/t/will-lfs-related-functionality-come-to-hf-api/146721 | Foro HF (Wauplin) | 1 |
| 3 | Bad request for commit endpoint (foro) | https://discuss.huggingface.co/t/bad-request-for-commit-endpoint/65535 | Foro HF | 1 |
| 4 | Upload files to the Hub (docs oficiales v1.7.1) | https://huggingface.co/docs/huggingface_hub/v1.7.1/en/guides/upload | Doc oficial | 1 |
| 5 | huggingface_hub v1.0: Five Years (blog) | https://huggingface.co/blog/huggingface-hub-v1 | Blog oficial | 1 |
| 6 | HfHub.Commit (Elixir, hexdocs) | https://hexdocs.pm/hf_hub/HfHub.Commit.html | Referencia minimal externa | 1 |
| 7 | Model upload and versioning (TheNeuralBase) | https://theneuralbase.com/huggingface-api/learn/advanced/model-upload-and-versioning/ | Tutorial avanzado | 1 |
| 8 | Hub API Endpoints (docs) | https://huggingface.co/docs/hub/api | Doc oficial | 1 |
| 9 | OpenAPI spec HF Hub | https://huggingface.co/spaces/huggingface/openapi | Doc oficial | 1 |
| 10 | Storage limits (HF Hub docs) | https://huggingface.co/docs/hub/main/en/storage-limits | Doc oficial | 1 |
| 11 | HF Storage Buckets docs | https://huggingface.co/docs/hub/storage-buckets | Doc oficial | 1 |
| 12 | `_commit_api.py` (huggingface_hub source) | https://github.com/huggingface/huggingface_hub/blob/v1.7.0/src/huggingface_hub/_commit_api.py | Source code | 1 |
| 13 | `lfs.py` (huggingface_hub source) | https://github.com/huggingface/huggingface_hub/blob/0b55fb46/src/huggingface_hub/lfs.py | Source code | 1 |
| 14 | `_upload_large_folder.py` | https://github.com/huggingface/huggingface_hub/blob/0b55fb46/src/huggingface_hub/_upload_large_folder.py | Source code | 1 |
| 15 | `cli/upload.py` | https://github.com/huggingface/huggingface_hub/blob/0b55fb46/src/huggingface_hub/cli/upload.py | Source code | 1 |
| 16 | `hf_api.py` (v0.25.0.rc0) | https://github.com/huggingface/huggingface_hub/blob/v0.25.0.rc0/src/huggingface_hub/hf_api.py | Source code | 1 |
| 17 | PR #888 — ✨ New create_commit API | https://github.com/huggingface/huggingface_hub/pull/888 | PR histórica | 1 |
| 18 | PR #920 — chunked preupload | https://github.com/huggingface/huggingface_hub/pull/920 | PR histórica | 1 |
| 19 | PR #1117 — NDJSON multipart commit | https://github.com/huggingface/huggingface_hub/pull/1117 | PR clave | 1 |
| 20 | PR #1699 — preupload lfs files before commit | https://github.com/huggingface/huggingface_hub/pull/1699 | PR | 1 |
| 21 | Issue #786 — Support python 3.10 (deprecate 3.6) | https://github.com/huggingface/huggingface_hub/issues/786 | Issue | 1 |
| 22 | Issue #946 — Uploading empty file fails | https://github.com/huggingface/huggingface_hub/issues/946 | Issue | 1 |
| 23 | Issue #2491 — most effective folder upload | https://github.com/huggingface/huggingface_hub/issues/2491 | Issue | 1 |
| 24 | About releases (GitHub docs) | https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases | Doc oficial | 1 |
| 25 | GitHub Releases billing discussion | https://github.com/orgs/community/discussions/73875 | Discusión | 1 |
| 26 | GitHub with Cloudflare R2 storage | https://mzfit.app/blog/github_with_cloudflare_r2/ | Blog técnico | 1 |
| 27 | Backblaze B2 for AI/ML storage | https://backblazeb2.ai/ | Página oficial | 1 |
| 28 | Backblaze B2 con boto3 | https://www.backblaze.com/docs/cloud-storage-use-the-aws-sdk-for-python-with-backblaze-b2 | Doc oficial | 1 |
| 29 | OmniBioAI Model Registry | https://github.com/man4ish/omnibioai-model-registry | Repo (patrón manifest) | 1 |
| 30 | TinyMLDelta: incremental ML updates embedded | https://medium.com/@felixgalindo91/introducing-tinymldelta-incremental-ml-model-updates-for-tiny-devices-96663edd1991 | Blog técnico | 1 |
| 31 | ML files too big for GitHub (BunnyCDN) | https://blog.aris.moe/posts/large-ml-files-and-hosting/ | Blog técnico | 1 |
| 32 | Upload File to Git LFS using REST API | https://github.com/orgs/community/discussions/24623 | Discusión | 1 |
| 33 | Git LFS overview: upload flow | https://www.mslinn.com/git/5100-git-lfs-overview.html | Tutorial | 1 |
| 34 | DeepWiki Hub API and Programmatic Access | https://deepwiki.com/huggingface/hub-docs/3-hub-api-and-programmatic-access | Wiki generada | 1 |
| 35 | Hugging Face / Dragonfly (CNCF) | https://www.cncf.io/blog/2023/11/16/hugging-face-accelerates-distribution-of-models-and-datasets-based-on-dragonfly/ | Blog técnico | 1 |
| 36 | PR #920 SemanticDiff | https://app.semanticdiff.com/gh/huggingface/huggingface_hub/pull/920/overview | Diff visual | 1 |
| 37 | Upload guide v0.13.0.rc0 | https://huggingface.co/docs/huggingface_hub/v0.13.0.rc0/en/guides/upload | Doc histórica | 1 |
| 38 | HfApi Client v0.19.3 | https://huggingface.co/docs/huggingface_hub/v0.19.3/package_reference/hf_api | Doc histórica | 1 |

### Gaps de evidencia

- **MCP `youtube` (fase 1b):** 0 resultados con query enfocada. Tema demasiado nicho para charlas/tutoriales en video. No se justificó bajar `aai_threshold` para activar AAI fallback (no había videos candidato con `priority_score` suficiente).
- **Discusiones específicas Py3.6 en HF Hub Discord/Slack interno:** sin acceso. La búsqueda en GitHub Issues público + foro fue exhaustiva pero no cubre canales privados.
- **Validación empírica con un POST real al endpoint correcto:** no se ejecutó en esta ronda — se difiere a la fase de implementación (próximo paso).
