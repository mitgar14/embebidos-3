# Atomic swap del engine TRT y recovery post-crash (V-2, MVP embebidos-3)

Dominio: estrategia de atomicidad del swap engine TRT en el builder del proyecto **embebidos-3** (Jetson Nano B01, JetPack 4.6.1 = Linux 4.9, Python 3.6.9, ext4 sobre SD card, sustentación académica en vivo).

Origen de la investigación: bug **V-2** documentado en `agent-docs/VACIOS.md`. El swap actual en `scripts/nano_build_engine.sh:172-184` es una secuencia de dos `mv` consecutivos no atómicos entre sí:

```bash
rm -f "$PREV_ENGINE" "$PREV_META"
if [[ -f "$ACTIVE_ENGINE" ]]; then
    mv "$ACTIVE_ENGINE" "$PREV_ENGINE"           # punto 1
    [[ -f "$ACTIVE_META" ]] && mv "$ACTIVE_META" "$PREV_META" || true
fi
mv "$STAGING_ENGINE" "$ACTIVE_ENGINE"            # punto 2
```

Ventana vulnerable: SIGKILL (OOM), reboot brusco o corte de luz exactamente entre punto 1 y punto 2 deja `engines/best_fp16.engine` inexistente, aunque `.previous/best_fp16.engine.old` y `.staging/best_fp16.engine.new` están válidos. El server arranca con `engine_binary_present=false` → estado `no_model`. El recovery actual (`scripts/recover_job_state.py`) solo limpia `active_job.json` huérfano; no inspecciona coherencia entre los tres paths.

---

## Ronda 1 — 2026-05-16 21:30:00 (medio)

### Pregunta de partida

¿Cuál de las cuatro opciones planteadas (A recovery extendido con sentinel · B symlink swap · C `renameat2 RENAME_EXCHANGE` · D aceptar riesgo) es la más sustentada por la comunidad ML production + embedded A/B, considerando el contexto MVP académico con Py3.6 sobre ext4 en SD card?

### Track A — agentes de research

#### A.1 · `research-code` — patrón hot-swap industria ML + embedded A/B

**Veredicto central:** ninguno de los 7 proyectos analizados (Triton, TF Serving, vLLM, KServe, Cog, Ray Serve, OSTree, Mender, swupdate) usa `RENAME_EXCHANGE`. El patrón canónico es **write-to-temp + rename(2) + sentinel file**.

**Patrones identificados (4 tipos):**

1. **Versioned directories con lifecycle en RAM** (TF Serving, Triton).
   - Subdirectorios numerados `model_name/1/`, `model_name/2/`. Polling detecta el de versión más alta.
   - Carga nueva en RAM manteniendo vieja activa hasta alcanzar `kReady`.
   - Atomicidad delegada a la máquina de estados en proceso, NO al filesystem.
   - Fuentes: `tensorflow_serving/sources/storage_path/file_system_storage_path_source.cc:73-164`, `tensorflow_serving/core/aspired_versions_manager.cc:140-280` (TF Serving). `src/model_repository_manager/model_repository_manager.cc:900-1007` (Triton core).
   - Recovery post-crash: simplemente re-poll del filesystem al restart.

2. **Symlink temporal + rename(2) como flip atómico** (OSTree, Cog).
   - **OSTree** (`src/libostree/ostree-sysroot-deploy.c:2179-2227`): función `prepare_new_bootloader_link` escribe `boot/loader.tmp` apuntando al nuevo deployment, llama `syncfs()`, después `swap_bootloader` hace `glnx_renameat(boot_fd, "loader.tmp", boot_fd, "loader")`. Comentario en línea 2223: *"Renaming now should give us atomic semantics"*.
   - **Cog** (`pkg/weights/lockfile/lockfile.go`): `atomicWriteFile` escribe a path temporal y rename al destino final. Centinelas `.cog/ready`, `.cog/failed`, `.cog/downloading` en el directorio de pesos. El runtime **bloquea el `setup()` hasta que el centinela `.ready` existe**.
   - Este es el patrón POSIX clásico: `rename(2)` es atómico para el observador, y el commit es la última operación.

3. **A/B con bootloader flags y estado persistente** (Mender, swupdate).
   - Variables clave: `upgrade_available`, `mender_boot_part`, `bootcount`. Persistidas en U-Boot environment (escritura redundante con checksum, atómica a nivel bootloader).
   - **Late commit**: el nuevo software hace `upgrade_available=0` recién cuando arranca bien. Si muere antes, `bootcount` supera `bootlimit` → rollback automático.
   - No aplicable a nuestro caso (no controlamos el bootloader del Nano).

4. **In-process rolling update con réplicas concurrentes** (vLLM, KServe, Ray Serve).
   - vLLM: `pause_generation` + `reload_weights` en RAM + `resume_generation`. Swap en GPU/CPU RAM, no en disco.
   - KServe/Knative: nueva Revision = nuevo pod con `/mnt/models` fresco. Tráfico redirigido por Knative.
   - Ray Serve: rolling de réplicas con drain de requests en vuelo.
   - No aplicable: requiere múltiples instancias o GPU con suficiente VRAM. El Nano tiene 4 GB compartidos.

**Por qué NADIE usa `RENAME_EXCHANGE` (cuatro razones documentadas):**

1. **Sin binding C estándar hasta glibc 2.28 (2018)**. Llamarlo requiere `syscall(__NR_renameat2, ...)` directo o librería auxiliar.
2. **No disponible en todos los filesystems**: ext4/XFS/Btrfs sí, FAT32 no, vfat no hasta kernel 6.0. OSTree explícitamente evita usarlo en `/boot` cuando es vfat.
3. **El truco del symlink ya es suficiente y más portable**. La secuencia write-to-temp + rename(2) tiene las mismas garantías POSIX y funciona en cualquier filesystem.
4. **No resuelve el problema de la pre-condición**: ambos paths deben existir, falla `ENOENT` si no. No elimina la necesidad de tener un staging válido antes de la transición.

**Recomendación del agente para embebidos-3: patrón centinela atómico + rename(2)**, inspirado en Cog `.cog/ready` y OSTree `loader.tmp`. Plan en dos capas:

- **Capa 1 (builder)**: después de los `mv` existentes, escribir un centinela `$ACTIVE_META.ready` con `echo + mv`. El rename del centinela es la última operación atómica que marca commit.
- **Capa 2 (recovery)**: `recover_job_state.py` verifica existencia del centinela `.ready` al startup. Si falta + hay `.previous` con su propio `.ready` → auto-promote.

Estimación: **<20 LOC totales**, sin nuevas dependencias, sin renameat2, sin reestructurar el swap. Patrón canónico industria validado por Cog (Replicate) + OSTree (Fedora CoreOS).

#### A.2 · `research-web` — atomicity, crash safety, SD cards, RENAME_EXCHANGE viability

**Distinción CRÍTICA: atomicidad observable vs. crash safety.**

`rename(2)` es atómico para el OBSERVADOR en sistema corriendo (garantía POSIX: "no point at which another process will find newpath missing"). Esa garantía **NO se extiende a power-loss crash**:

- **Dan Luu** (`danluu.com/file-consistency/`, análisis canónico): *"rename isn't atomic on crash. POSIX says that rename is atomic, but this only applies to normal operation, not to crashes."*
- **Pillai et al. OSDI '14**: estudio sistemático de inconsistencias post-crash en filesystems Linux.
- **Kernel ext4 admin-guide** (Mauro Carvalho Chehab, `infradead.org/~mchehab/kernel_docs/`): *"rename does not wait for this flush to complete, and therefore provides no atomicity guarantee — it is possible to end up with only partial new content after a crash."*

**En ext4 con `data=ordered` (default JetPack)**: el journal protege CONSISTENCIA DE METADATOS (no verás entries de directorio rotas), pero NO INTEGRIDAD DE DATOS — un crash a mitad puede dejar el archivo con cero bytes o contenido parcialmente viejo. La excepción es btrfs (rename crash-atomic), no ext4.

**Capa adicional de riesgo: SD card FTL.** Whitepaper de Technologic Systems "Preventing Filesystem Corruption in Embedded Linux" (Dec 2020): el controller hace wear leveling en allocation groups de ~4 MiB. Corte de luz durante GC del FTL puede corromper hasta 4 MiB independiente del kernel. El SD puede reportar success de `fsync()` sin haber persistido. Para eliminar este riesgo: eMMC en Data Reliability Mode o batería de respaldo.

**Patrón obligatorio para durabilidad** (PostgreSQL `durable_rename`):
```
fsync(fd_nuevo) → fsync(fd_target_existente_si_aplica) → rename(tmp, target) →
fsync(fd_renamed_file) → fsync(parent_dir_fd)
```
El `fsync(parent_dir)` es lo que la mayoría olvida — es lo que persiste la **entrada de directorio** del rename.

**`renameat2 RENAME_EXCHANGE` viabilidad técnica en el Nano:**

- ✅ Kernel 4.9 (JetPack 4.6.1) > 3.15 requerido. Soportado en ext4 desde 3.15.
- ✅ Syscall number `__NR_renameat2 = 276` en aarch64 (vía asm-generic, commit 63ba600).
- ✅ Paquete PyPI `renameat2` (jordemort/python-renameat2, MIT) requiere exactamente Py3.6 y hace syscall directo vía CFFI sin glibc 2.28.
- ⚠️ Glibc del JetPack 4.6.1 es <2.28 → no expone wrapper nativo; usar lib PyPI o ctypes directo.
- ⚠️ Instalar CFFI puede requerir compilación local en el Nano. Fricción adicional.
- ⚠️ **No es práctica industria**: ningún ML server lo usa, OSTree explícitamente lo evita.

**NVIDIA Forums sobre TRT engine hot-swap en Jetson Nano:**
- Hilo [212196](https://forums.developer.nvidia.com/t/when-to-update-a-tensorrt-engine-file/212196) (2022): el moderador NVES confirma que los engines no son portables y hay que reconstruirlos en el device. **No da patrón de hot-swap.**
- Hilo [323664](https://forums.developer.nvidia.com/t/switching-tensorrt-compiled-engines-without-reloading-from-file/323664) (2025): no se pudo extraer contenido completo en esta ronda (low-confidence).
- API `IRefitter` permite actualizar pesos sin reconstruir engine, pero solo si la arquitectura no cambia. No aplica a re-entrenamientos con nuevas clases.
- **Conclusión**: NVIDIA no documenta hot-swap atómico en Jetson Nano. Estamos solos en ese frente.

### Track B — búsqueda ampliada

#### Fase 1 — `discover.py` (Exa Search)

23 fuentes descubiertas. Prioridad para fase 2:
- Foundational LWN: [Articles/569134](https://lwn.net/Articles/569134/) (introducción `renameat2` por Corbet, Oct 2013), [574380](https://lwn.net/Articles/574380/) (cross-rename v2), [896359](https://lwn.net/Articles/896359/) (vfat support).
- Man pages oficiales: man7.org [rename(2)](https://www.man7.org/linux/man-pages/man2/rename.2.html), manpages.org [renameat2(2)](https://manpages.org/renameat2/2), [exch(1)](https://man7.org/linux/man-pages/man1/exch.1.html) de util-linux.
- Source code minimal de referencia: [util-linux/exch.c](https://github.com/util-linux/util-linux/blob/master/misc-utils/exch.c), [AbhyudayaSharma/exchange](https://github.com/AbhyudayaSharma/exchange), [rubenvannieuwpoort/atomic-exchange](https://github.com/rubenvannieuwpoort/atomic-exchange), [google/renameio](https://github.com/google/renameio/) (Go).
- **python-renameat2** ([jordemort](https://github.com/jordemort/python-renameat2), [PyPI](https://pypi.org/project/renameat2/), [docs](https://python-renameat2.readthedocs.io/en/latest/renameat2.html)) — wrapper Python 3.6+ vía CFFI.
- Casos reales de adopción: [coreos/bootupd #454](https://github.com/coreos/bootupd/issues/454) → PR #669 mergeada (Jun 2024) usando `local_exchange` de Rust openat. Caveat de cgwalters: el PR mergeado "made this safer but still not transactional" en caso de updates multi-directorio (no aplica a nuestro caso de un solo archivo).
- Discusiones tendencia: [LWN xfs atomic file content exchanges](https://lwn.net/Articles/969167/) (Linux 6.13+ explora primitivas más fuertes con `EXCHANGE_RANGE` ioctl + reflink).
- Tendencia opuesta: ClickHouse issue [#96835](https://github.com/ClickHouse/ClickHouse/issues/96835) — graceful fallback necesario cuando `RENAME_EXCHANGE` no soportado (filesystems de red, ZFS <2.2).

#### Fase 1b — MCP `youtube`

`youtube_search` con query combinada `atomic file rename linux ext4 crash safety renameat2 RENAME_EXCHANGE | model serving hot reload triton kserve | A/B update embedded ostree mender rauc`, `top_k=10`, `include_long_tail=true`, `lang_pref=en`, `diversity=0.5` → 200u quota.

**Resultado: 0 hits.** Igual que V-1, gap registrado: tema demasiado nicho técnico para charlas en video. No se justifica bajar `aai_threshold` para AAI fallback (no había candidatos con `priority_score` alto). Quota usada: 200u.

#### Fase 2 — Exa `crawling_exa` sobre 10 URLs prioritarias

**LWN 569134 (Corbet, Oct 2013)**: introducción canónica de `renameat2`. Confirma que `renameat()` es atómico para UN archivo (POSIX), pero swap de DOS archivos requería nueva primitiva. Miklos Szeredi propone `renameat2` con flag `RENAME_EXCHANGE`. Si flags=0, idéntico a `renameat()`. Si flags=RENAME_EXCHANGE: ambos archivos siguen existiendo, nombres intercambiados atomicamente. Main use case original: union filesystems / overlayfs whiteouts.

**man7 rename(2)**: confirma garantía atomicity observable (POSIX) y diferenciación entre tipos de error. Soporte ext4 desde Linux 3.15. `renameat2()` HISTORY: "Linux 3.15, glibc 2.28". `ENOENT` si `RENAME_EXCHANGE` y newpath no existe.

**python-renameat2 docs**: API expone `exchange(a, b)`, `rename(old, new, replace=, whiteout=)`, `renameat2(olddirfd, oldpath, newdirfd, newpath, flags)`. Doc literal: *"This is an atomic operation; that is to say, there is no possible intermediate state where the files could be 'partially' swapped; either the call succeeds and the files are exchanged, or the call fails and the files are not exchanged."*

**jordemort/python-renameat2 README**: 13 stars, 1 fork, MIT, último release v0.4.4 (Aug 2022). Crítico: *"This package requires Python 3.6"* ✓. *"Your kernel must be version 3.15.0 or newer"* ✓ (Nano tiene 4.9). *"This package does not have any libc requirements; glibc includes a wrapper for renameat2 in version 2.28 and newer, but this is significantly newer than the glibc in any of the manylinux containers. In order to avoid inflicting any libc requirements on the user, this package brings its own wrapper function that makes the system call directly."* ✓ (JetPack glibc <2.28).

**coreos/bootupd #454 timeline**: el issue lo abrió Javier Martinez (Mar 2023), mergeado PR #669 por HuijingHei (Jun 2024). Logic adoptada: `cp -a fedora .fedora.tmp` → apply diff a tmp → `RENAME_EXCHANGE(.fedora.tmp, fedora)` → remove tmp. Caveat post-merge (Nov 2024) de cgwalters: arquitectura multi-directorio (BOOT + fedora) hace que el flujo no sea totalmente transactional. Para un solo path (nuestro caso), sí es transactional.

**StackExchange canónico (Aarkon, A.B accepted 13 score)**: ejemplo práctico ctypes desde Py3 (arquitectura x86_64 específico):
```python
import ctypes
libc = ctypes.CDLL(None)
libc.syscall(316, -100, b"a.txt", -100, b"b.txt", 2)
# 316 = SYS_renameat2 x86_64, -100 = AT_FDCWD, 2 = RENAME_EXCHANGE
```
ilkkachu sugiere alternativa equivalente: symlink atómico `ln -sf nuevo current`.

### Síntesis: confrontación de las opciones

Reformulado tras la investigación. La opción A inicial (recovery extendido sin sentinel) era débil; el patrón canónico industria (Cog + OSTree) la mejora con un centinela:

| Criterio | A++ (centinela + recovery) | B (symlink swap) | C (RENAME_EXCHANGE) | D (aceptar riesgo + logging) |
|---|---|---|---|---|
| **LOC totales** | ~20 (3 en builder + 15-20 en recovery) | ~30 (cambio de paradigma: worker debe seguir symlink, swap del symlink, eliminar 2 paths intermedios) | ~10 con `pip install renameat2` o ~30 con ctypes raw + fallback | ~5 (solo logging post-mortem) |
| **Tiempo realista** | 60-90 min impl + 30 min validar | 2-3 h impl + 60 min validar | 60 min si pip funciona; +60-120 min si compilar CFFI en el Nano falla | 15 min |
| **Cobertura SIGKILL/OOM** | ✅ centinela escrito al final → si no existe, recovery promueve | ✅ symlink swap es atómico | ✅ syscall atómico | ❌ sigue vulnerable, solo loguea |
| **Cobertura power-loss** | ⚠️ limitada por journal ext4 + FTL SD (igual que todas) | ⚠️ idem | ⚠️ idem (no mejora vs A++) | ❌ no |
| **Dependencias nuevas** | 0 (Python stdlib + bash) | 0 | `renameat2` PyPI (CFFI, posible recompilación en el Nano) | 0 |
| **Patrón industria** | ✅ Cog `.cog/ready`, OSTree `loader.tmp` | ✅ OSTree usa esto (junto con sentinel) | ❌ NADIE en ML serving o embedded A/B lo usa | ❌ |
| **Pre-condición "primer build sin engine previo"** | ✅ funciona (no requiere ambos paths) | ✅ funciona (crea symlink si no existe) | ❌ ENOENT → necesita rama especial | ✅ |
| **Riesgo de regresión** | Bajo (solo agrega centinela al final; recovery es código nuevo aislado) | Medio (cambia paradigma, todos los consumidores del path deben seguir symlinks) | Medio (instalación CFFI + manejo errores syscall + fallback para FS no soportado) | Bajísimo |
| **Alineación CLAUDE.md MVP académico** | ✅ ✅ | ✅ | ⚠️ (overengineering para MVP) | ⚠️ (curita) |
| **Tendencia industria** | El sentinel pattern es default en ML serving production | OSTree lo combina con sentinel | LWN explora primitivas más fuertes en kernel 6.13+ (XFS atomic file content exchange ioctl) — RENAME_EXCHANGE es paso intermedio | n/a |

### Decisión recomendada (sustentada)

**Recomendación primaria: A++ híbrida (centinela `.ready` + recovery extendido).**

Justificación cruzada:

1. **Es el patrón canónico de la industria ML production.** Cog (Replicate, ~10k stars) usa exactamente esto en `pkg/weights/lockfile/lockfile.go`. OSTree (Fedora CoreOS) lo usa en `prepare_new_bootloader_link` + `swap_bootloader`. Ambos son código de producción con millones de deploys.

2. **El research-code confirma que `RENAME_EXCHANGE` no se usa en producción** por 4 razones documentadas (sin binding C estándar histórico, no portable a todos los FS, el truco symlink+rename es suficiente, no resuelve la pre-condición). El research-web confirma viabilidad técnica del syscall pero también explicita la fricción de empaquetarlo en Py3.6 del Nano (compilación CFFI).

3. **El research-web aclara la ilusión de "atomicidad crash-safe"**: ni `rename(2)` ni `RENAME_EXCHANGE` ni el symlink swap son verdaderamente crash-safe en ext4 sobre SD card. El journal protege metadatos, no datos. El FTL del SD puede romper todo. **Para SIGKILL/OOM (mucho más probable en el Nano que power-loss durante una demo), el journal sí protege, y todas las opciones A/B/C son equivalentes en ese sentido.** Lo que diferencia es la complejidad operativa y de impl.

4. **El centinela `.ready` añade un beneficio que ninguna primitiva atómica da**: el server puede VERIFICAR antes de leer. Si `.ready` no existe pero el `.engine` sí, sabemos que el build fue interrumpido — independiente de qué primitiva use el filesystem. Es defensa en profundidad sobre la atomicidad del FS.

5. **Para "primer build sin engine previo"**, A++ funciona sin código especial. C requiere rama explícita por `ENOENT`.

6. **Alineación CLAUDE.md**: *"busca construir un MVP. Las implementaciones que veas que son complicadas realmente están orientadas a ofrecer el funcionamiento más fluido durante la sustentación"*. A++ es minimal, sin nuevas deps, sin compilación CFFI, sin syscall raw. Aplicable inmediatamente.

**Descartar B (symlink swap)**: equivalentemente atómico que A++ en práctica, pero requiere cambiar el paradigma de paths del proyecto (worker TRT, dashboard, server endpoints, todos los consumidores deberían seguir symlinks correctamente). Mayor superficie de regresión para igual ganancia.

**Descartar C (RENAME_EXCHANGE)**: técnicamente válido pero no usado por nadie en la industria por razones documentadas. La fricción de empaquetar CFFI en Py3.6 del Nano es real. No resuelve nada que A++ no resuelva, y agrega complejidad. **No es la dirección a la que va la industria** — los proyectos modernos optan por sentinel o symlink, no por primitivas exóticas del kernel.

**Descartar D (aceptar riesgo)**: el usuario explicitó "fundamentos para todas las decisiones". Solo loguear sin remediar no es decisión fundamentada — es resignación.

**Detalle de implementación A++ propuesto:**

*Builder (`scripts/nano_build_engine.sh:172-184`, reemplazar el bloque del swap):*

```bash
# 11. swap atómico (V-2 fix: centinela .ready como marker de commit)
rm -f "$PREV_ENGINE" "$PREV_META" "$PREV_DIR/best_fp16.engine.ready"
if [[ -f "$ACTIVE_ENGINE" ]]; then
    mv "$ACTIVE_ENGINE" "$PREV_ENGINE"
    [[ -f "$ACTIVE_META" ]] && mv "$ACTIVE_META" "$PREV_META" || true
    # Marcar el .previous como válido (rename atómico del centinela)
    if [[ -f "$PREV_ENGINE" ]]; then
        echo "{\"committed_at\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" \
            > "$PREV_DIR/best_fp16.engine.ready.tmp"
        mv "$PREV_DIR/best_fp16.engine.ready.tmp" "$PREV_DIR/best_fp16.engine.ready"
    fi
fi
# Borrar el centinela del active mientras está incompleto
rm -f "$ROOT/engines/best_fp16.engine.ready"
mv "$STAGING_ENGINE" "$ACTIVE_ENGINE"
# fsync del archivo recién promovido (durabilidad)
python3 -c "import os; fd=os.open('$ACTIVE_ENGINE', os.O_RDONLY); os.fsync(fd); os.close(fd)"
# Commit: rename atómico del centinela como ÚLTIMA operación
echo "{\"committed_at\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"engine_sha256\":\"$(sha256sum "$ACTIVE_ENGINE" | awk '{print $1}')\"}" \
    > "$ROOT/engines/best_fp16.engine.ready.tmp"
mv "$ROOT/engines/best_fp16.engine.ready.tmp" "$ROOT/engines/best_fp16.engine.ready"
# fsync del directorio padre (PostgreSQL durable_rename pattern)
python3 -c "import os; fd=os.open('$ROOT/engines', os.O_RDONLY); os.fsync(fd); os.close(fd)"
```

*Recovery (`scripts/recover_job_state.py`, agregar función):*

```python
def reconcile_engine_state():
    """Auto-promueve .previous si el active quedó sin .ready.
    Aplica V-2 fix: si el builder murió entre los dos mv del swap,
    .previous tiene .ready pero engines/best_fp16.engine no existe.
    """
    active_engine = Path('/home/jetson/embebidos-3/engines/best_fp16.engine')
    active_ready = active_engine.with_suffix('.engine.ready')
    prev_engine = Path('/home/jetson/embebidos-3/engines/.previous/best_fp16.engine.old')
    prev_ready = prev_engine.with_suffix('.engine.old.ready')

    if active_engine.exists() and active_ready.exists():
        return  # estado consistente
    if not active_engine.exists() and prev_engine.exists() and prev_ready.exists():
        # promover previous a active
        os.rename(str(prev_engine), str(active_engine))
        os.rename(str(prev_ready), str(active_ready))
        # opcional: log degraded
        write_recovery_log({
            'action': 'auto_promoted_previous',
            'reason': 'active_engine missing .ready, prev had valid .ready',
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })
```

Llamar `reconcile_engine_state()` después de `recover_active_job_if_any()` en el startup del server.

---

## Ronda 2 — 2026-05-16 23:00:00 (medio, confirmatoria)

### Pregunta de partida

¿La recomendación A++ de ronda 1 resiste contraevidencia dirigida? ¿Hay race conditions, edge cases, postmortems o tendencias industria que invaliden o requieran ajustar la decisión?

**Justificación del usuario para la ronda 2** (registrada en memory `feedback-decisiones-tecnicas.md`): *"decisiones técnicas avanzadas se confirman con exploración+fundamentación técnica; no con lo que yo prefiera (ya que yo no sé)"*. La ronda 1 produjo una recomendación; la ronda 2 la estresa antes de implementar.

### Track A — agentes de research

#### A.1 · `research-code` — contraevidencia código en repos ML production + embedded

**Conclusión del agente: A++ confirmada con un ajuste de precisión.**

Hallazgos cruzados:

1. **`replicate/cog` está EXPANDIENDO el patrón**, no abandonándolo. PR [#2974](https://github.com/replicate/cog/pull/2974) (2026-05-01, "feat: experimental managed weights") implementa exactamente la misma estrategia que A++ propone, descrita literal en `specs/draft-weights.md`: *"a `ready` marker MUST NOT appear until all weight data is fully written and flushed to disk"*, escrito vía tempfile+fsync+rename. Commit [18052e3c9b](https://github.com/replicate/cog/commit/18052e3c9b) (2026-05-05) refinó `atomicWriteFile` cerrando edge-case de `defer tmp.Close()`. Mantenimiento activo, no deprecación.
2. **TF Serving y Triton — gap de ronda 1 CONFIRMADO con precisión**. DeepWiki sobre `tensorflow/serving` confirmó: descubren versiones por parsing numérico de directorios (`GetChildren()` + parse), señal "ready" es interna al proceso (`ServableState` en RAM). DeepWiki sobre `triton-inference-server/core`: usan timestamps de directorios (`ModelTimestamp`). **No usan sentinel file porque su arquitectura es proceso único con estado en RAM**. Caso embebido (builder Bash + server Python, comunicación solo por filesystem) es estructuralmente diferente y justifica el centinela.
3. **Migración DE centinela A otro patrón: NINGUNA encontrada**. Queries `gh search prs "deprecate sentinel"`, `"remove .ready"`, `"replace marker file"` → 0 resultados. OSTree commits recientes a `ostree-sysroot-deploy.c` van en dirección de mejorar `syncfs()` (2025-08-18), alineados con la filosofía del patrón, no contra.
4. **Race condition reader-during-swap (mostlygeek/llama-swap [#667](https://github.com/mostlygeek/llama-swap/issues/667), abril 2026): NO aplica al caso embebido**. El race de llama-swap ocurre porque hay múltiples goroutines concurrentes operando sobre el mismo proceso (hot-reload del modelo en RAM). En embebidos-3, el server Python carga el engine UNA vez al startup (`nano_server.py`), sin hot-reload. No hay concurrencia real entre builder y server.
5. **`mcu-tools/mcuboot` [#1966](https://github.com/mcu-tools/mcuboot/issues/1966) (power failure durante escritura del magic) es el análogo más cercano en firmware embebido**. Solución adoptada: escribir el magic en una sola operación atómica. En A++, el centinela `.ready` se escribe vía `echo + mv` (rename atómico observable), lo que es suficiente para el propósito de señalización en ext4 local.

**Ajuste recomendado por research-code**: `reconcile_engine_state()` debe VALIDAR que `.previous/best_fp16.engine.old` existe como archivo (no asumir que el directorio basta). Si el builder muriera entre crear `.previous/` y escribir el engine, el directorio existiría vacío.

Fuentes ronda 2 (nuevas, no duplicadas de ronda 1):

- [replicate/cog PR #2974](https://github.com/replicate/cog/pull/2974) — managed weights con ready marker
- [replicate/cog commit 18052e3c9b](https://github.com/replicate/cog/commit/18052e3c9b) — refinamiento atomicWriteFile
- [DeepWiki tensorflow/serving sentinel query](https://deepwiki.com/search/does-tensorflow-serving-use-an_d589ff04-546a-4960-97a5-69284a9991de)
- [DeepWiki triton-inference-server/core sentinel query](https://deepwiki.com/search/does-triton-inference-server-u_1978ebd8-dd7b-4345-998a-1c5611a97f11)
- [mostlygeek/llama-swap #667](https://github.com/mostlygeek/llama-swap/issues/667) — race condition swap (no aplica)
- [mcu-tools/mcuboot #1966](https://github.com/mcu-tools/mcuboot/issues/1966) — power failure magic (analógico embebido)
- [gpustack/gpustack #2869](https://github.com/gpustack/gpustack/issues/2869) — lock file stale (problema opuesto a A++)

#### A.2 · `research-web` — contraevidencia race, TOCTOU, postmortems, primitivas nuevas

**Conclusión del agente: A++ confirmada con 2 ajustes obligatorios.**

Hallazgos cruzados:

1. **TOCTOU del centinela existe estructuralmente pero ACOTADO al boot**. Issues encontrados: [PrefectHQ/fastmcp #3938](https://github.com/PrefectHQ/fastmcp/issues/3938) (abril 2026, race en reload concurrente con read), [lemonade-sdk/lemonade #1603](https://github.com/lemonade-sdk/lemonade/issues/1603) (abril 2026, TOCTOU en serving). En embebidos-3, `reconcile_engine_state()` corre SOLO en startup del server cuando el builder ya terminó. **No hay acceso concurrente real**. NO refuta A++.
2. **AJUSTE OBLIGATORIO #1 — fsync explícitos**: Ferrite (ASPLOS 2016), LevelDB [#195](https://github.com/google/leveldb/issues/195), FastForward [#386](https://github.com/strawgate/fastforward/issues/386), y [Unix SE 464382](https://unix.stackexchange.com/questions/464382) convergen: en ext4 con `auto_da_alloc` (default JetPack), `rename` fuerza flush de bloques pero NO espera confirmación. Sin `fsync(engine_fd)` antes del mv + `fsync(parent_dir_fd)` después, un crash puede dejar el engine en destino con contenido parcial. Ronda 1 citó PostgreSQL `durable_rename` como modelo pero NO especificó estos fsyncs en el pseudocódigo. **Implementación requerida**:

```python
# Antes del mv (en builder)
with open(staging_engine, 'rb') as f:
    os.fsync(f.fileno())
# mv
os.rename(staging_engine, active_engine)
# Después del mv: fsync del directorio padre
parent_fd = os.open(os.path.dirname(active_engine), os.O_RDONLY)
os.fsync(parent_fd)
os.close(parent_fd)
```

3. **AJUSTE OBLIGATORIO #2 — threat model FTL acotado a power-loss**: ronda 1 documentó genéricamente "SD FTL puede romper atomicidad". Refinamiento de [Hackaday](https://hackaday.com/2016/08/03/single-board-revolution-preventing-flash-memory-corruption/) + [RPi forums](https://forums.raspberrypi.com/viewtopic.php?t=326237) (respuesta de ingeniero de RPi): bajo SIGKILL o reboot normal, el kernel completa transacciones del journal antes de desmontar; los file descriptors se cierran limpiamente. **La corrupción del FTL es exclusiva de power-loss real (corte de luz físico)**, no de SIGKILL/OOM. Esto simplifica el threat model: para SIGKILL bastan los fsyncs del ajuste #1. Para power-loss, la mitigación es operativa (UPS o batería) — fuera de scope del fix de software.
4. **Primitivas nuevas post-2020 en kernel: no aplicables**. `xfs atomic file updates` ioctl ([LWN 2021](https://lwn.net/ml/linux-fsdevel/161723932606.3149451.12366114306150243052.stgit@magnolia/)) está en Linux 6.13+. SquirrelFS y CXL PM aplican a persistent memory, no a ext4/SD. Kernel 4.9 del JetPack 4.6.1 NO tiene primitivas más fuertes que rename+fsync.
5. **Sin postmortems específicos del patrón centinela en Cog/OSTree/Fedora desde 2024**. Búsqueda exhaustiva: ausencia justificada de contraevidencia, no evidencia inversa.

Fuentes ronda 2 (nuevas):

- [PrefectHQ/fastmcp #3938](https://github.com/PrefectHQ/fastmcp/issues/3938) — race reload-during-read
- [lemonade-sdk/lemonade #1603](https://github.com/lemonade-sdk/lemonade/issues/1603) — TOCTOU model load
- [LevelDB #195](https://github.com/google/leveldb/issues/195) — fsync y metadata reordering
- [strawgate/fastforward #386](https://github.com/strawgate/fastforward/issues/386) — fsync(dir) faltante
- [Unix SE 464382](https://unix.stackexchange.com/questions/464382) — Ferrite ASPLOS 2016
- [coreos/fedora-coreos-tracker #1938](https://github.com/coreos/fedora-coreos-tracker/issues/1938) — Zincati rollback (no aplica)
- [LWN xfs atomic file updates RFC v3](https://lwn.net/ml/linux-fsdevel/161723932606.3149451.12366114306150243052.stgit@magnolia/)
- [Hackaday SD card corruption](https://hackaday.com/2016/08/03/single-board-revolution-preventing-flash-memory-corruption/)
- [RPi forums eMMC random power downs](https://forums.raspberrypi.com/viewtopic.php?t=326237)

### Track B — Skipped justificado

**Decisión**: omitir Track B (discover.py + youtube + Exa crawling) en ronda 2. Justificación:

- Los dos agentes de Track A ya cubrieron las fuentes que `discover.py` habría descubierto (GitHub issues, foros, DeepWiki, papers académicos via Semantic Scholar y Exa semantic search).
- El gap del MCP youtube documentado en ronda 1 ("0 hits, tema demasiado nicho técnico para charlas en video") sigue vigente; reintentar gastaría 200u quota sin valor esperado.
- Crawling Exa adicional sería redundante con lo que los agentes ya leyeron.
- Alineado con principio MVP académico (CLAUDE.md): no exhaustividad por exhaustividad.

Si el spike empírico (`.planning/spikes/`) revela findings inesperados, se justifica una ronda 3 dirigida.

### Síntesis cruzada: A++ confirmada con 3 ajustes

| Ajuste | Origen | Implementación requerida |
|---|---|---|
| **#1: fsync explícitos** | research-web | `fsync(engine_fd)` antes del `mv` + `fsync(parent_dir_fd)` después. Aplicar en TODOS los `mv` del swap (active→previous, staging→active, y la escritura del centinela). |
| **#2: threat model acotado** | research-web | Documentar en el `.md` y en el código que A++ cubre SIGKILL/OOM/reboot. Power-loss requiere UPS/batería (fuera de scope software). |
| **#3: validar contenido `.previous`** | research-code | `reconcile_engine_state()` verifica `prev_engine.is_file()` (no `prev_dir.exists()`) antes de promover. Rechaza directorios vacíos. |

**Sin contraevidencia significativa** del patrón centinela en sí. La decisión A++ resiste el estrés.

### Decisión final ronda 2

A++ **CONFIRMADA con los 3 ajustes obligatorios**. La implementación productiva derivada del Spike 002 (siguiente fase) DEBE incorporar los tres ajustes. La validación empírica en hardware real es el próximo paso (`.planning/spikes/`) — esta ronda cierra el gap bibliográfico, el spike cierra el empírico.

---

## Ronda 2 — Spike empírico (2026-05-16 16:30-16:42 UTC-5)

Validación empírica de A++ en hardware real del Nano (kernel 4.9.337-tegra, ext4 sobre SD card). Sandbox `/home/jetson/spike-v2/` aislado. Artefactos completos en `.planning/spikes/`.

### Tres spikes ejecutados

| Spike | Verdict | Asserts | Hallazgo |
|---|---|---|---|
| 001 reproducir-v2-baseline | **VALIDATED ✓** | 4/4 PASS | V-2 reproducido empíricamente: active engine missing, .previous con engine válido (sha256 matches pre-race), .staging intacto. Confirma el bug exactamente como ronda 1 lo modeló. |
| 002 aplus-recovery-resuelve-v2 | **VALIDATED ✓** | 5/5 PASS | A++ con los 3 ajustes resuelve V-2: `reconcile.py` detectó V-2, auto-promovió previous, active recuperado con sha256 integridad confirmada (`64c9c55f9bf5398f...`), recovery <200ms. |
| 003 aplus-no-false-positives-integridad | **VALIDATED ✓** | 8/8 PASS (3 escenarios) | A: estado consistente → no_op (idempotencia OK). B: previous engine sin .ready → degraded (centinela necesario). C: previous .ready sin engine → degraded (ajuste #3 validó contenido, no solo existencia). |

### Implementación validada empíricamente

El código de A++ que el Spike 002+003 validó (throwaway, sandbox) tiene **3 archivos clave** que deben portarse a producción:

**1. Builder swap con A++ (`scripts/nano_build_engine.sh` fase 10, líneas 191-204 actuales)**

Reemplazar el bloque de swap actual por:

```bash
# Fase 10: swap atómico A++ (V-2 fix con 3 ajustes de ronda 2)
fsync_path "$STAGING_ENGINE"                   # Ajuste #1: fsync staging antes del mv
rm -f "$PREV_ENGINE" "$PREV_META" "$PREV_READY"
if [[ -f "$ACTIVE_ENGINE" ]]; then
    mv "$ACTIVE_ENGINE" "$PREV_ENGINE"
    [[ -f "$ACTIVE_META" ]] && mv "$ACTIVE_META" "$PREV_META" || true
    if [[ -f "$ACTIVE_READY" ]]; then
        mv "$ACTIVE_READY" "$PREV_READY"       # .ready hereda validez del active anterior
    fi
    fsync_path "$PREV_DIR"                     # Ajuste #1: fsync parent dir
fi
rm -f "$ACTIVE_READY"
mv "$STAGING_ENGINE" "$ACTIVE_ENGINE"
fsync_path "$ACTIVE_ENGINE"                    # Ajuste #1: fsync engine promovido
# Commit: rename atómico del centinela como ÚLTIMA operación
TMP_READY="${ACTIVE_READY}.tmp"
ACTIVE_SHA=$(sha256sum "$ACTIVE_ENGINE" | awk '{print $1}')
echo "{\"committed_at\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"engine_sha256\":\"$ACTIVE_SHA\"}" > "$TMP_READY"
fsync_path "$TMP_READY"
mv "$TMP_READY" "$ACTIVE_READY"
fsync_path "$ENGINES_DIR"                      # Ajuste #1: fsync parent dir final
```

Donde `fsync_path()` es un helper Bash que invoca Python (Bash no tiene fsync builtin):

```bash
fsync_path() {
    python3 -c "
import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
try: os.fsync(fd)
finally: os.close(fd)
" "$1"
}
```

**2. Recovery en `scripts/recover_job_state.py`** — agregar función `reconcile_engine_state()`:

```python
def reconcile_engine_state(root=None):
    """A++ recovery: si active engine missing + previous válido con .ready, auto-promueve.
    Ajustes de ronda 2: #1 fsync(parent_dir), #2 cubre SIGKILL/OOM (no power-loss),
    #3 valida prev_engine.is_file() (no solo directorio)."""
    root = root or "/home/jetson/embebidos-3"
    engines_dir = Path(root) / "engines"
    active_engine = engines_dir / "best_fp16.engine"
    active_meta = engines_dir / "best_fp16.engine.meta.json"
    active_ready = engines_dir / "best_fp16.engine.ready"
    prev_dir = engines_dir / ".previous"
    prev_engine = prev_dir / "best_fp16.engine.old"
    prev_meta = prev_dir / "best_fp16.engine.old.meta.json"
    prev_ready = prev_dir / "best_fp16.engine.old.ready"

    if active_engine.is_file() and active_ready.is_file():
        return {"action": "no_op", "reason": "active engine + .ready coherent"}

    if (not active_engine.is_file() and prev_engine.is_file() and prev_ready.is_file()):
        os.rename(str(prev_engine), str(active_engine))
        if prev_meta.is_file():
            os.rename(str(prev_meta), str(active_meta))
        os.rename(str(prev_ready), str(active_ready))
        fd = os.open(str(engines_dir), os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
        return {"action": "auto_promoted_previous", "reason": "V-2 recovery"}

    return {"action": "degraded", "reason": "active missing AND previous invalid",
            "active_engine_exists": active_engine.is_file(),
            "prev_engine_exists": prev_engine.is_file(),
            "prev_ready_exists": prev_ready.is_file()}
```

**3. Integración en `scripts/nano_server.py`** — llamar `reconcile_engine_state()` después de `recover_job_state()` en el startup.

### Conclusión del spike empírico

**A++ con los 3 ajustes de ronda 2 está EMPÍRICAMENTE VALIDADA para implementación productiva.**

Evidencia agregada de las dos rondas + spike:
- Bibliográfica (ronda 1): 53 fuentes, patrón canónico industria
- Bibliográfica (ronda 2): contraevidencia dirigida nula, 3 ajustes técnicos identificados
- Empírica (spike): 17/17 asserts PASS, integridad sha256 confirmada, hardware real

Recovery ventana cubierta: SIGKILL/OOM entre primer y segundo `mv` del swap (caso V-2 estricto) Y entre segundo `mv` y escritura del centinela del active (caso degraded detectable).

Recovery ventana NO cubierta: power-loss físico (out of scope software, mitigación operativa). Race entre los 2 `mv` del previous (~µs, degraded detectable).

---

## Historial de investigación

| Ronda | Fecha | Profundidad | Foco |
|---|---|---|---|
| 1 | 2026-05-16 | medio | V-2: opciones A/B/C/D para fix del swap no atómico; sustento industria ML production + embedded A/B + Linux kernel internals |
| 2 | 2026-05-16 | medio (confirmatoria) | V-2: contraevidencia dirigida del patrón centinela. Resultado: A++ confirmada + 3 ajustes (fsync explícitos, threat model acotado a power-loss, validar contenido `.previous`). Track B skipped justificadamente. |

---

## Fuentes consultadas (acumulado)

| # | Título | URL | Tipo | Ronda |
|---|---|---|---|---|
| 1 | LWN · Exchanging two files (Corbet, intro renameat2) | https://lwn.net/Articles/569134/ | LWN canon | 1 |
| 2 | LWN · cross rename v2 | https://lwn.net/Articles/574380/ | LWN | 1 |
| 3 | LWN · vfat RENAME_EXCHANGE support | https://lwn.net/Articles/896359/ | LWN | 1 |
| 4 | LWN · first iteration of rename2 support | https://lwn.net/Articles/606237/ | LWN | 1 |
| 5 | LWN · xfs atomic file content exchanges (L6.13+) | https://lwn.net/Articles/969167/ | LWN | 1 |
| 6 | LWN · renameat2 GIT PULL Miklos Szeredi → Linus | https://lwn.net/Articles/592952/ | LWN kernel mailing | 1 |
| 7 | LWN · A way to do atomic writes | https://lwn.net/Articles/789600/ | LWN | 1 |
| 8 | man7 · rename(2) Linux manual page | https://www.man7.org/linux/man-pages/man2/rename.2.html | Doc oficial kernel | 1 |
| 9 | manpages.org · renameat2(2) | https://manpages.org/renameat2/2 | Doc oficial | 1 |
| 10 | man7 · exch(1) (util-linux) | https://man7.org/linux/man-pages/man1/exch.1.html | Doc oficial | 1 |
| 11 | kdave.github.io · Atomic cross-rename of two paths | https://kdave.github.io/atomic-cross-rename/ | Blog técnico | 1 |
| 12 | Blog Javier Martinez · Atomically exchange vfat files in Linux | https://blog.dowhile0.org/2024/02/10/atomically-exchange-vfat-files-in-linux/ | Blog técnico | 1 |
| 13 | Oracle blog · XFS – Atomic File Content Commit in Linux 6.13 | https://blogs.oracle.com/linux/xfs-atomic-file-content-commit-in-linux-613 | Blog técnico | 1 |
| 14 | Dan Luu · Files are hard (file consistency canon) | https://danluu.com/file-consistency/ | Blog técnico canon | 1 |
| 15 | Unix SE · Which filesystems require fsync for crash-safety with rename | https://unix.stackexchange.com/questions/464382/ | Foro técnico | 1 |
| 16 | Unix SE · swap files instead of cp with temp file (canónica) | https://unix.stackexchange.com/questions/673591/swap-files-instead-of-cp-with-temp-file | Foro técnico | 1 |
| 17 | GitHub Issue OSTEP · rename atomic crash | https://github.com/remzi-arpacidusseau/ostep-code/issues/10 | Foro académico | 1 |
| 18 | infradead/Mauro · ext4 General Information admin-guide | https://www.infradead.org/~mchehab/kernel_docs/admin-guide/ext4.html | Doc oficial kernel | 1 |
| 19 | Embedded TS · Preventing Filesystem Corruption in Embedded Linux | https://www.embeddedts.com/assets/preventing-filesystem-corruption-in-embedded-linux | Whitepaper | 1 |
| 20 | Red Hat · Possible Data loss on ext4 after power loss | https://access.redhat.com/solutions/369383 | Doc Red Hat | 1 |
| 21 | kernel.org · commit vfs: add renameat2 syscall (520c8b16) | https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=520c8b16505236fc82daa352e6c5e73cd9870cff | Código kernel | 1 |
| 22 | GitHub · asm-generic Add renameat2 (NR=276 aarch64) | https://github.com/torvalds/linux/commit/63ba600028a001fa19f427486527387f54926d61 | Código kernel | 1 |
| 23 | PyPI · renameat2 wrapper | https://pypi.org/project/renameat2/ | Lib PyPI | 1 |
| 24 | python-renameat2 docs | https://python-renameat2.readthedocs.io/en/latest/renameat2.html | Doc lib | 1 |
| 25 | GitHub · jordemort/python-renameat2 | https://github.com/jordemort/python-renameat2 | Repo lib | 1 |
| 26 | Gist · Python renameat2 RENAME_EXCHANGE wrapper ctypes | https://gist.github.com/dbnicholson/ac5e299a9f18663ba5b11674ee5aaf39 | Código | 1 |
| 27 | discuss.python.org · Extending os.rename with swapping | https://discuss.python.org/t/extending-os-rename-to-support-file-swapping-and-whiteout/22257 | Foro Python devs | 1 |
| 28 | GitHub · util-linux/exch.c (canonical CLI) | https://github.com/util-linux/util-linux/blob/master/misc-utils/exch.c | Código C | 1 |
| 29 | GitHub · AbhyudayaSharma/exchange | https://github.com/AbhyudayaSharma/exchange | Código C | 1 |
| 30 | GitHub · rubenvannieuwpoort/atomic-exchange | https://github.com/rubenvannieuwpoort/atomic-exchange | Código C | 1 |
| 31 | GitHub · google/renameio (Go) | https://github.com/google/renameio/ | Lib Go | 1 |
| 32 | GitHub · coreos/bootupd #454 (caso real adoption) | https://github.com/coreos/bootupd/issues/454 | Issue + PR | 1 |
| 33 | GitHub · ClickHouse #96835 (fallback RENAME_EXCHANGE) | https://github.com/ClickHouse/ClickHouse/issues/96835 | Issue prod | 1 |
| 34 | GitHub · hanwen/go-fuse #398 (rename atomicity) | https://github.com/hanwen/go-fuse/issues/398 | Issue | 1 |
| 35 | GitHub · ostreedev/ostree src/libostree/ostree-sysroot-deploy.c | https://github.com/ostreedev/ostree/blob/1d2b902b/src/libostree/ostree-sysroot-deploy.c | Source code canon | 1 |
| 36 | OSTree · Atomic Upgrades docs | https://ostreedev.github.io/ostree/atomic-upgrades/ | Doc oficial | 1 |
| 37 | GitHub · replicate/cog pkg/weights/lockfile/lockfile.go | https://github.com/replicate/cog/blob/main/pkg/weights/lockfile/lockfile.go | Source code canon | 1 |
| 38 | GitHub · tensorflow/serving file_system_storage_path_source.cc | https://github.com/tensorflow/serving/blob/14a87232/tensorflow_serving/sources/storage_path/file_system_storage_path_source.cc | Source code | 1 |
| 39 | GitHub · tensorflow/serving aspired_versions_manager.cc | https://github.com/tensorflow/serving/blob/14a87232/tensorflow_serving/core/aspired_versions_manager.cc | Source code | 1 |
| 40 | GitHub · triton-inference-server/core model_repository_manager.cc | https://github.com/triton-inference-server/core/blob/e6fcb0c7/src/model_repository_manager/model_repository_manager.cc | Source code | 1 |
| 41 | GitHub · vllm-project/vllm | https://github.com/vllm-project/vllm | Repo | 1 |
| 42 | GitHub · kserve/kserve | https://github.com/kserve/kserve | Repo | 1 |
| 43 | GitHub · ray-project/ray (Ray Serve) | https://github.com/ray-project/ray | Repo | 1 |
| 44 | GitHub · mendersoftware/mender | https://github.com/mendersoftware/mender | Repo | 1 |
| 45 | Mender docs · Architecture Overview (A/B partition) | https://docs.mender.io/2.1/architecture/overview | Doc oficial | 1 |
| 46 | Mender blog · Robust OTA updates with A/B Partitions | https://mender.io/blog/robust-ota-updates-with-partitions-for-linux-devices | Blog oficial | 1 |
| 47 | HN · Mender developer rollback bootcount | https://news.ycombinator.com/item?id=13745959 | HN thread | 1 |
| 48 | RAUC · Using RAUC | https://rauc.readthedocs.io/en/latest/using.html | Doc oficial | 1 |
| 49 | GitHub · sbabic/swupdate | https://github.com/sbabic/swupdate | Repo | 1 |
| 50 | PostgreSQL · durable_rename reference | https://ryogrid.github.io/create_pg_super_document/d/durable_rename.html | Ref académica | 1 |
| 51 | NVIDIA Forums · When to update a TRT engine file | https://forums.developer.nvidia.com/t/when-to-update-a-tensorrt-engine-file/212196 | Foro oficial NVIDIA | 1 |
| 52 | NVIDIA Forums · Switching TRT engines without reload | https://forums.developer.nvidia.com/t/switching-tensorrt-compiled-engines-without-reloading-from-file/323664 | Foro oficial NVIDIA | 1 |

### Gaps de evidencia

**Tras ronda 2:**

- **MCP `youtube`**: ratificado como nicho sin cobertura en video. Cerrado definitivamente.
- **NVIDIA Forums [323664](https://forums.developer.nvidia.com/t/switching-tensorrt-compiled-engines-without-reloading-from-file/323664)**: contenido pendiente; ronda 2 confirmó que A++ no depende de esto.
- **Validación empírica del centinela bajo SIGKILL real**: **EN EJECUCIÓN** en `.planning/spikes/` (Spike 001 reproducir baseline, Spike 002 A++ recovery resuelve V-2, Spike 003 no false positives + integridad sha256). Resultados se documentarán en sección "Ronda 2 — Spike empírico" al completarse.
- **Power-loss físico del Nano**: fuera de scope software (mitigación operativa con UPS/batería). Ronda 2 acotó el threat model a SIGKILL/OOM/reboot — único escenario donde A++ tiene cobertura técnica.
- **Postmortems internos de Replicate sobre Cog `.cog/ready` en producción**: no son públicos; búsqueda exhausta. Bajo confianza pero patrón validado por adopción activa (PR #2974, mayo 2026).
