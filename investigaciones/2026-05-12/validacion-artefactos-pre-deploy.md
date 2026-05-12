# Validación de artefactos `.tflite` / `.onnx` pre-deploy Jetson Nano B01

**Proyecto:** `embebidos-3` (clasificador de residuos Jetson Nano B01, entrega 2026-05-26).
**Dominio:** gates obligatorios desde Vast.ai antes de transferir el `.tflite` (Track A) o el `.onnx` (Track B) al Jetson Nano. Cierre del gap drop INT8 en Maxwell `sm_53` (D14).
**Documentos hermanos:** [`decisiones-D1-D15-ledger.md`](decisiones-D1-D15-ledger.md) (D8, D12, D13, D14, D15) · [`compatibilidad-stack-cloud-jetson.md`](compatibilidad-stack-cloud-jetson.md) · [`infraestructura-training-vastai-uv-hf.md`](infraestructura-training-vastai-uv-hf.md) · [`dataset-roboflow-yolov8.md`](dataset-roboflow-yolov8.md).
**Fecha de cierre:** 2026-05-12.

---

## 1. Resumen ejecutivo

El Jetson Nano B01 con JetPack 4.6.1 tiene un runtime **inmutable** (Python 3.6, TFLite 2.5, TensorRT 8.2.1.8, Maxwell `sm_53` sin Tensor Cores INT8 ni instrucción `dp4a`). Cualquier artefacto generado en Vast.ai (CUDA 12.4, TF 2.15, Ultralytics 8.4.46) debe validarse contra ese runtime **antes de transferirlo**, para evitar el ciclo "bajar al Nano → falla por `op_version` inválido → reentrenar 2 horas → repetir".

Cuatro gates obligatorios resumen las decisiones D12, D13 + el plan B D15 y el corolario empírico D14:

| Gate | Aplica | Decisión | Razón |
|------|--------|----------|-------|
| **Gate 1 — TFLite `op_version`** | Track A | **D12** | Inspeccionar el flatbuffer con `tflite==2.5.0` y verificar que ningún operador exceda la versión máxima del runtime 2.5 (`CONV_2D` ≤ 5, `DEPTHWISE_CONV_2D` ≤ 4, etc.). |
| **Gate 2 — TFLite carga test** | Track A | **D12** | Cargar el `.tflite` con `tflite_runtime==2.5.0.post1` (wheel Coral CP38 x86, idéntico runtime al Nano). Si falla con `Didn't find op for builtin opcode`, abortar o aplicar plan B. |
| **Gate 3 — ONNX ops blacklist TRT 8.2** | Track B | **D13** | Inspección del grafo ONNX contra la lista soportada por TRT 8.2-GA. Las ops `GridSample`, `DFT`, `IsInf`, `IsNaN`, `MelWeightMatrix`, `STFT`, `SequenceInsert`, `CumSum` NO están en TRT 8.2; `ConstantOfShape` solo en FP32. |
| **Gate 4 — Polygraphy + TRT 8.2.1 en Docker NGC** | Track B | **D13** | Ejecutar `polygraphy run --trt --onnxrt` en el container `nvcr.io/nvidia/tensorrt:21.11-py3` (TRT 8.2.1.8 idéntico al JetPack 4.6.1). `pip install polygraphy tensorrt` en Vast.ai CUDA 12.4 instala TRT 10+, NO 8.2 — Docker NGC obligatorio. |

Adicionalmente:

- **Plan B `TFLite_Detection_PostProcess` (D15):** si el wheel NVIDIA del Nano no incluye el custom op, fallback al wheel Coral CP36 aarch64 (URL + sha256 documentado en §3.3).
- **D14 FP16-only por default en Track B:** Maxwell `sm_53` carece de `dp4a` (introducida en Pascal `sm_61`) → speedup INT8 estructuralmente nulo. Experimento INT8 opcional 45–60 min en el Nano con criterio binario.

---

## 2. Mecánica del versioning TFLite (Gate 1 fundamento)

### 2.1 Por qué importa `op_version`

Cada operador TFLite tiene una versión máxima registrada en el runtime. Si el converter (TF 2.15 export) genera un `op_version` superior al runtime (TFLite 2.5 del Nano), el modelo lanza al cargar:

```
ERROR: Didn't find op for builtin opcode 'CONV_2D' version '5'.
An older version of this builtin might be supported.
Are you using old TFLite binary with newer model?
```

El **schema flatbuffer** (`TFLITE_SCHEMA_VERSION = 3`) es estable entre TF 2.5 y TF 2.21 (verificado en `tensorflow/lite/version.h` HEAD = TF 2.21 master). La compatibilidad real depende de **op versioning por operador**, no del schema container.

### 2.2 Histórico de issues confirmados

Tres reports cruzados confirman que el escenario es real:

- Issue [`tensorflow/tensorflow#41943`](https://github.com/tensorflow/tensorflow/issues/41943) (`mgalgs`, 2020): *"ERROR: Didn't find op for builtin opcode 'CONV_2D' version '5'."* Modelo generado con `tf-nightly 2.4`, fallo en runtime 2.2/2.3.
- Issue [`#50652`](https://github.com/tensorflow/tensorflow/issues/50652) (`djbacad`, TF 2.5.0 Python 3.6.9, 2021): *"Quantized Version of tf-lite model returning `ERROR: Didn't find op for builtin opcode 'CONV_2D' version '5'`."*
- Issue [`#43232`](https://github.com/tensorflow/tensorflow/issues/43232) (`juanpbotero98`, 2020): *"I'm trying to run inference of a custom trained `mobilenet_v2_coco17_320x320_tpu-8` model on a Raspberry pi [...] `Didn't find op for builtin opcode 'CONV_2D' version '5'`."*

El patrón es: **converter moderno genera op version N**, **runtime viejo soporta hasta versión N-1**. La mitigación es forzar el converter legacy (D12 flags) o downgrade manual del flatbuffer.

### 2.3 Tabla de `op_version` por riesgo (TF 2.15 INT8 PTQ → runtime TFLite 2.5)

| Op | Versión máx TFLite 2.5 | Versión generada TF 2.15 INT8 PTQ | Riesgo | Mitigación si falla |
|----|------------------------|------------------------------------|--------|----------------------|
| `CONV_2D` | 5 | 5 con activaciones estándar | **Bajo** (al límite) | Converter legacy (D12) o `flatbuffer_utils` downgrade |
| `DEPTHWISE_CONV_2D` | 4 | 4–5 según flags | **Medio** (verificación obligatoria) | Converter legacy (D12) |
| `FULLY_CONNECTED` | 4 | 4 para MobileNet v2 | Bajo | — |
| `QUANTIZE` / `DEQUANTIZE` | 2 | 2 | Bajo | — |
| `PAD`, `ADD`, `MUL`, `RESHAPE`, `CONCATENATION` | 1–2 | 1–2 | Ninguno | — |
| `TFLite_Detection_PostProcess` | custom | custom | Depende del wheel runtime | Ver §3.3 (Plan B Coral wheel) |
| `Cast v2+` | (requiere TF 2.7+) | Probablemente v1 si NMS embebido evita Cast | Bajo | — |
| `BatchMatMul v5+` | (requiere TF 2.6+) | No aplica en MV2 SSD plain | Ninguno | — |

**GAP residual:** las versiones exactas de `DEPTHWISE_CONV_2D` y `FULLY_CONNECTED` que TF 2.15 genera con INT8 PTQ desde TFOD API **no están documentadas públicamente**. La inspección flatbuffer post-conversión (§3.1) es obligatoria.

### 2.4 Por qué no existe flag `target_runtime_version`

El converter TFLite no tiene flag `--target_runtime_version` ni `min_runtime_version`. Los únicos flags que reducen la probabilidad de versiones altas son los del converter legacy:

```python
converter.experimental_new_quantizer = False  # cuantizador legacy
converter.experimental_new_converter  = False # converter TOCO legacy
```

**Tradeoff:** el converter legacy puede producir cuantización de menor calidad (entre +1 y +3 pp de drop adicional según Karimov et al. 2025), pero garantiza compatibilidad con runtimes antiguos.

---

## 3. Gate 1 + Gate 2 — Validación TFLite Track A (D12, D15)

### 3.1 Gate 1 — Inspección de `op_version` con paquete `tflite==2.5.0`

```bash
source /opt/venv/tracka/bin/activate
pip install tflite==2.5.0  # PyPI: package "tflite" (no "tflite-runtime")
```

```python
# scripts/validate_tflite_ops.py (o celda del notebook)
import tflite.Model
from pathlib import Path

MODEL = Path("track_a/exports/model_int8.tflite")

with open(MODEL, "rb") as f:
    buf = bytearray(f.read())

model = tflite.Model.Model.GetRootAsModel(buf, 0)
sg = model.Subgraphs(0)
op_codes = [model.OperatorCodes(i) for i in range(model.OperatorCodesLength())]

# Tabla de versiones máximas soportadas por runtime 2.5
MAX_VERSION_TFLITE_25 = {
    1: 5,   # CONV_2D
    4: 4,   # DEPTHWISE_CONV_2D
    9: 4,   # FULLY_CONNECTED
    # 114, 115 = QUANTIZE/DEQUANTIZE v2 OK
    # 1, 2, 3, 22, 41 = PAD/ADD/MUL/RESHAPE/CONCATENATION v1-2 OK
}

errors = []
for i in range(sg.OperatorsLength()):
    op = sg.Operators(i)
    code = op_codes[op.OpcodeIndex()]
    builtin = code.BuiltinCode()
    version = code.Version()
    max_v = MAX_VERSION_TFLITE_25.get(builtin)
    if max_v is not None and version > max_v:
        errors.append(f"Op {i}: builtin={builtin} version={version} > max {max_v}")
    print(f"Op {i}: builtin={builtin} version={version}")

if errors:
    print("\n❌ ERRORES op_version > runtime 2.5 max:")
    for e in errors:
        print(f"   {e}")
    print("\nAplicar flags D12 o flatbuffer_utils downgrade (§3.5).")
    raise SystemExit(1)
else:
    print(f"\n✅ Todos los operadores son compatibles con TFLite runtime 2.5")
```

### 3.2 Gate 2 — Carga test con wheel Coral CP38 x86

El wheel Coral CP38 x86 es **el mismo runtime TFLite 2.5** que el Nano usaría (wheel Coral CP36 aarch64), pero compilado para Vast.ai x86. Permite validar sin tener acceso físico al Nano.

```bash
# Wheel verificado disponible en Google Coral PyPI repo
pip install \
  "https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp38-cp38-linux_x86_64.whl"
```

```python
import tflite_runtime.interpreter as tflite

try:
    interp = tflite.Interpreter("track_a/exports/model_int8.tflite")
    interp.allocate_tensors()

    in_d = interp.get_input_details()
    out_d = interp.get_output_details()
    print(f"✅ Runtime 2.5 acepta el modelo")
    print(f"   Input dtype: {in_d[0]['dtype']}, shape: {in_d[0]['shape']}")
    # Si INT8: in_d[0]['dtype'] debe ser np.uint8, no np.float32

    # Validar que TFLite_Detection_PostProcess está embebido (4 outputs)
    assert len(out_d) == 4, \
        f"PTQ TFLite con NMS embebido debe tener 4 outputs (boxes/classes/scores/num_detections), got {len(out_d)}"
    print(f"   Outputs: {len(out_d)} tensores (boxes, classes, scores, num_detections)")
except ValueError as e:
    if "TFLite_Detection_PostProcess" in str(e):
        print("❌ FALLO: custom op TFLite_Detection_PostProcess no encontrado.")
        print("   Aplicar plan B D15 (§3.3) en el Nano.")
    else:
        raise
```

### 3.3 `TFLite_Detection_PostProcess` y plan B Coral wheel CP36 aarch64 (D15)

#### 3.3.1 Naturaleza del custom op

`TFLite_Detection_PostProcess` no es un Select TF op; es un **op nativo compilado** en `tensorflow/lite/kernels/detection_postprocess.cc`. **Debería estar incluido** en cualquier build completo de TFLite (no en builds minimal `tflite-micro` o similares).

> *"The `tflite_runtime` package is a fraction the size of the full `tensorflow` package and includes the bare minimum code required to run inferences with LiteRT — primarily the `Interpreter` Python class."*
> — Google AI Edge docs, 2025.

#### 3.3.2 GAP del wheel NVIDIA

**No existe documentación pública** que confirme verbatim si el wheel NVIDIA `tensorflow==2.5.0+nv21.8` (preinstalado en JetPack 4.6.1) incluye `TFLite_Detection_PostProcess` compilado. La arquitectura es distinta al wheel Coral: NVIDIA compiló TF completo con GPU/CUDA; Coral compiló solo el runtime TFLite. Inferencia: **debería estar pero sin garantía**.

#### 3.3.3 Plan B Coral wheel CP36 aarch64 (D15)

Si el wheel NVIDIA en el Nano lanza `Didn't find custom op TFLite_Detection_PostProcess`, fallback al wheel oficial Coral verificado disponible:

```
URL:    https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp36-cp36m-linux_aarch64.whl
sha256: 7c58b1a9fb2d2b24d6f0b0f8629ede7d288358e2cb93c68c3e4f78fd0ee7d1df
```

Listado verbatim en [`google-coral.github.io/py-repo/tflite-runtime/`](https://google-coral.github.io/py-repo/tflite-runtime/).

**Configuración exacta del Nano:** Python 3.6, aarch64, JetPack 4.6.1 → coincide con `cp36-cp36m-linux_aarch64`.

#### 3.3.4 Uso en el Nano

```bash
# En el Nano (post-deploy, si plan B requerido)
wget -q https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp36-cp36m-linux_aarch64.whl
echo "7c58b1a9fb2d2b24d6f0b0f8629ede7d288358e2cb93c68c3e4f78fd0ee7d1df  tflite_runtime-2.5.0.post1-cp36-cp36m-linux_aarch64.whl" | sha256sum -c -
pip3 install tflite_runtime-2.5.0.post1-cp36-cp36m-linux_aarch64.whl
```

```python
# inference_nano.py (en el Nano)
import tflite_runtime.interpreter as tflite  # NO `from tensorflow.lite`
interp = tflite.Interpreter("model_int8.tflite")
interp.allocate_tensors()
# resto del pipeline de inferencia
```

#### 3.3.5 Repos de referencia con build TF JetPack 4.6.1

- [`Qengineering/TensorFlow-JetsonNano`](https://github.com/Qengineering/TensorFlow-JetsonNano) — wheels TF para Jetson Nano (incluye TFLite).
- [`PINTO0309/Tensorflow-bin`](https://github.com/PINTO0309/Tensorflow-bin) — wheels alternativos con XNNPACK + Multi-Threads.
- [`xuhj-code/Tensorflow-bin`](https://github.com/xuhj-code/Tensorflow-bin) — fork con custom ops MediaPipe.

### 3.4 Flags del converter (export Track A)

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)

# Conversión INT8 con representative dataset
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.uint8
converter.inference_output_type = tf.uint8

# CRÍTICO (D12): flags conservadores para minimizar op_version
converter.experimental_new_quantizer = False
converter.experimental_new_converter  = False

tflite_model = converter.convert()
Path("track_a/exports/model_int8.tflite").write_bytes(tflite_model)
```

### 3.5 Workaround `flatbuffer_utils.py` (si Gate 1 falla)

Archivo [`tensorflow/lite/tools/flatbuffer_utils.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/flatbuffer_utils.py) permite parsear y reescribir flatbuffers. **No existe script oficial de downgrade de versión**, pero es técnicamente posible bajar manualmente el campo `version` de cada operador.

**Frágil, no documentado oficialmente.** Solo aplicar si:

1. Gate 1 detecta `op_version` excedido.
2. Re-export con D12 flags no resolvió.
3. Re-train con dataset distinto no resolvió (caso patológico).

```python
# Patrón (esqueleto, requiere adaptación)
import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb

# Cargar
with open("model_int8.tflite", "rb") as f:
    buf = bytearray(f.read())
model = schema_fb.ModelT.InitFromBuf(buf, 0)

# Modificar (e.g., bajar CONV_2D v5 → v4 si runtime acepta v4)
for op_code in model.operatorCodes:
    if op_code.builtinCode == 1 and op_code.version > 4:  # CONV_2D
        print(f"Downgrade CONV_2D v{op_code.version} -> v4")
        op_code.version = 4

# Re-serializar
builder = flatbuffers.Builder(1024)
builder.Finish(model.Pack(builder), b"TFL3")
with open("model_int8_downgraded.tflite", "wb") as f:
    f.write(bytes(builder.Output()))
```

**Riesgo:** el campo `version` del op_code es solo metadata; las características reales del operador no cambian. Si TF 2.15 genera CONV_2D con features de v5 (e.g., parámetro `depthwise_multiplier` o `dilation`), bajar version a 4 hace que el runtime intente parsearlo como v4 y falle de forma distinta.

**Veredicto:** workaround último recurso. Prefiere re-export con D12 flags o re-train.

---

## 4. Gate 3 — Inspección ONNX ops blacklist TRT 8.2 (Track B, D13)

### 4.1 Lista soportada por TRT 8.2-GA

Recuperada verbatim de [`onnx-tensorrt/docs/operators.md release/8.2-GA`](https://github.com/onnx/onnx-tensorrt/blob/release/8.2-GA/docs/operators.md):

> *"TensorRT 8.2 supports operators up to Opset 13."*

Tabla parcial relevante para YOLOv8n:

| Op ONNX | Soportado TRT 8.2 | Tipos | Restricciones |
|---------|--------------------|-------|---------------|
| `Add` | Sí | FP32, FP16, INT32 | — |
| `Concat` | Sí | FP32, FP16, INT32, INT8, BOOL | — |
| **`ConstantOfShape`** | Sí | **FP32 únicamente** | Riesgo si onnxslim genera tipos INT64 |
| `Conv` | Sí | FP32, FP16, INT8 | 2D o 3D, pesos como initializer |
| `Equal` | Sí | FP32, FP16, INT32 | — |
| `Gather` / `GatherND` | Sí | FP32, FP16, INT8, INT32 | Issue [`#4383`](https://github.com/NVIDIA/TensorRT/issues/4383) Gather rank-0 con opsets ≥17; opset 11 lo evita |
| **`GridSample`** | **❌ AUSENTE** | — | Solo en YOLOv8-seg, no detección |
| `NonMaxSuppression` | Sí `[EXPERIMENTAL]` | FP32, FP16 | Inputs como initializers; evitar con `nms=False` |
| `Range` | Sí | FP32, FP16, INT32 | Inputs flotantes solo como initializers |
| `Reshape` | Sí | FP32, FP16, INT32, INT8, BOOL | — |
| `Resize` | Sí | FP32, FP16 | Modos: `half_pixel`, `pytorch_half_pixel`, `tf_half_pixel_for_nn`, `asymmetric`, `align_corners`; interpolación: `nearest`, `linear` |
| `ScatterND` | Sí | FP32, FP16, INT8, INT32 | — |
| `Sigmoid` / `HardSigmoid` | Sí | FP32, FP16, INT8 | — |
| `Softmax` | Sí | FP32, FP16 | — |
| `Upsample` | Sí | FP32, FP16 | — |
| `Where` | Sí | FP32, FP16, INT32, BOOL | — |

### 4.2 Blacklist TRT 8.2 (ops NO soportadas, evitar)

| Op | Por qué bloquear |
|----|------------------|
| `GridSample` | **No aparece en operators.md** (sí en TRT 8.4+). Solo YOLOv8-seg lo usa |
| `DFT`, `IsInf`, `IsNaN` | Operaciones numéricas avanzadas; no soportadas en TRT 8.2 |
| `MelWeightMatrix`, `STFT` | Audio/señales, no aplicable |
| `SequenceInsert` | Operadores de secuencia (NLP); no aplicable |
| `CumSum` | Suma acumulada; no soportada en TRT 8.2 (sí en 8.4+) |
| `NonZero` | No soportada |
| `RoiAlign` | Solo segmentación; no aplica YOLOv8 detect |
| `QLinearConv`, `QLinearMatMul` | Solo INT8 ONNX (QDQ flow), no aplica |
| `Reciprocal` | No soportada (improbable en YOLOv8n FP32 estático) |

### 4.3 Análisis YOLOv8n detección pura con flags actuales

Con el export canónico de Track B (D13 + `dataset-roboflow-yolov8.md` §"export"):

```python
model.export(format="onnx", imgsz=416, opset=11,
             simplify=True, dynamic=False, nms=False)
```

- `GridSample` aparece **solo en YOLOv8-seg**, no en detección. ✅ No problema.
- `NonMaxSuppression` **NO está en el grafo** con `nms=False`. ✅
- `ConstantOfShape` con tipo no-FP32 → **riesgo bajo** si `dynamic=False` (genera initializers estáticos). Validación requerida.
- `Resize` con `mode='linear'` o `'nearest'` → soportado por opset 11.

### 4.4 Script de inspección Gate 3

```bash
source /opt/venv/trackb/bin/activate
pip install onnx
```

```python
# scripts/validate_onnx_ops.py
import onnx
from pathlib import Path

MODEL = Path("track_b/exports/best.onnx")

m = onnx.load(str(MODEL))

# Verificar IR y opset
assert m.opset_import[0].version == 11, f"Opset {m.opset_import[0].version} ≠ 11"
assert m.ir_version <= 10, f"IR version {m.ir_version} > 10 puede romper onnxslim 0.6.x / TRT 8.2"
print(f"✅ Opset: {m.opset_import[0].version}, IR: {m.ir_version}")

# Inspeccionar ops
ops = sorted({n.op_type for n in m.graph.node})
print(f"Ops en el grafo: {ops}")

BLACKLIST_TRT82 = {
    "GridSample", "DFT", "IsInf", "IsNaN",
    "MelWeightMatrix", "STFT",
    "SequenceInsert", "CumSum",
    "NonZero", "RoiAlign",
    "QLinearConv", "QLinearMatMul",
    "Reciprocal",
}
hits = set(ops) & BLACKLIST_TRT82
if hits:
    print(f"❌ Ops problemáticas TRT 8.2 encontradas: {hits}")
    raise SystemExit(1)
else:
    print("✅ Ninguna op de la blacklist TRT 8.2 presente.")

# Verificar ConstantOfShape NO usa tipos exóticos
for n in m.graph.node:
    if n.op_type == "ConstantOfShape":
        for a in n.attribute:
            if a.name == "value":
                dt = a.t.data_type
                # ONNX TensorProto.FLOAT == 1
                if dt != 1:
                    print(f"❌ ConstantOfShape con dtype {dt} (debe ser 1=FP32)")
                    raise SystemExit(1)
                print(f"✅ ConstantOfShape dtype: {dt} (FP32)")

# Verificar NMS no embebido (debería ser False con nms=False)
nms_ops = [n for n in m.graph.node if n.op_type == "NonMaxSuppression"]
if nms_ops:
    print(f"⚠️  NonMaxSuppression encontrado ({len(nms_ops)} nodos). Verificar flags del export (nms=False).")
else:
    print("✅ Sin NonMaxSuppression embebido (NMS se hará en CPU NumPy en Nano).")

print("\n✅ Gate 3 OK — modelo listo para Gate 4 (Polygraphy).")
```

---

## 5. Gate 4 — Polygraphy + TRT 8.2.1 en Docker NGC (Track B, D13)

### 5.1 Por qué Docker NGC y no `pip install`

**Crítico:** si en el container Vast.ai (CUDA 12.4) corres `pip install polygraphy tensorrt`, obtienes **TRT 10+, no 8.2**. Para validar contra el TRT exacto del Nano (8.2.1.8), **es obligatorio** usar el Docker NGC `21.11-py3` que tiene TRT 8.2.1 + CUDA 11.5 + Ubuntu 20.04.

### 5.2 Imágenes NGC TensorRT candidatas

| Imagen NGC | Versión TensorRT | CUDA | Ubuntu | Python | Coincide JetPack 4.6.1 |
|------------|-------------------|------|--------|--------|------------------------|
| `nvcr.io/nvidia/tensorrt:21.10-py3` | 8.0.x | 11.4 | 20.04 | 3.8 | No |
| **`nvcr.io/nvidia/tensorrt:21.11-py3`** | **8.2.1** | 11.5 | 20.04 | 3.8 | ✅ **Sí (versión exacta)** |
| `nvcr.io/nvidia/tensorrt:22.01-py3` | 8.2.3 | 11.5 | 20.04 | 3.8 | Aproximado (no exacto) |

### 5.3 Polygraphy 0.49.x

[`NVIDIA/TensorRT/tools/Polygraphy/CHANGELOG.md`](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/CHANGELOG.md) — versión actual `v0.49.27` (2025).

> *"Fixed a bug where `explicit_batch` would be provided by default on TRT 10.0, where it has been removed."*
> — v0.49.5, 2024-01-16.

Esto confirma que **polygraphy 0.49.x funciona con TRT 8 Y TRT 10**. No existe serie 0.50+. PyPI lista hasta `0.49.26`.

### 5.4 Comando de validación canónico

```bash
# 1. Pull imagen NGC (~6 GB, una sola vez)
docker pull nvcr.io/nvidia/tensorrt:21.11-py3

# 2. Validación completa: parser ONNX + comparación numérica TRT vs ORT
docker run --rm --gpus all \
  -v "$(pwd)":/workspace \
  nvcr.io/nvidia/tensorrt:21.11-py3 \
  bash -c "
    pip install -q polygraphy onnx &&
    polygraphy run /workspace/track_b/exports/best.onnx \
      --onnxrt --trt \
      --atol 1e-2 --rtol 1e-2 \
      --input-shapes images:[1,3,416,416]
  "
```

**Flags clave:**

- `--onnxrt` — corre el modelo con ONNX Runtime (referencia).
- `--trt` — corre con TensorRT 8.2.1 dentro del container.
- `--atol 1e-2 --rtol 1e-2` — tolerancia absoluta y relativa para comparar outputs (suficiente para FP16; FP32 usaría 1e-5).
- `--input-shapes images:[1,3,416,416]` — shape fijo. Para dynamic shapes usar formato `--input-shapes images:[min,opt,max]`.

### 5.5 Alternativas (informativo)

```bash
# Solo validación ligera del parser (sin Docker, instala TRT 10 inútil para Nano)
pip install onnx onnxruntime polygraphy
polygraphy inspect model track_b/exports/best.onnx --display-as=trt

# trtexec dentro del Docker NGC (alternativa a polygraphy)
docker run --rm --gpus all \
  -v "$(pwd)":/workspace \
  nvcr.io/nvidia/tensorrt:21.11-py3 \
  trtexec --onnx=/workspace/track_b/exports/best.onnx \
          --shapes=images:1x3x416x416 \
          --fp16 \
          --verbose 2>&1 | grep -E "ERROR|WARNING|Parsing|Building"
```

### 5.6 Limitación arquitectural — x86 + TRT 8.2 ≠ aarch64 Maxwell `sm_53`

Este gate es **necesario pero no suficiente**:

| Aspecto | x86 + TRT 8.2 vía Docker | aarch64 `sm_53` Nano real |
|---------|--------------------------|----------------------------|
| Validación parser ONNX | ✅ | ✅ |
| Detección ops fuera de opset 13 | ✅ | ✅ |
| Comparación numérica `--onnxrt` | ✅ | ✅ |
| Tiempos reales en Maxwell | ❌ | ✅ |
| Fusiones de kernels específicas Maxwell | ❌ | ✅ |
| Comportamiento INT8 calibrador (sin TC) | ❌ | ✅ |

**Implicación:** después de pasar Gate 4 en Vast.ai, ejecutar una **corrida rápida** en el Nano (10–15 min de smoke test) antes del primer ciclo de training completo. Compilar el `.engine` con:

```bash
# En el Nano (post-deploy, primer smoke test)
trtexec --onnx=best.onnx \
        --fp16 \
        --workspace=1024 \
        --saveEngine=best.engine \
        --verbose 2>&1 | tee trt_build.log

# Validar latencia
trtexec --loadEngine=best.engine \
        --shapes=images:1x3x416x416 \
        --iterations=100
# Esperar latencia ~30-50 ms = 20-33 FPS
```

### 5.7 Polygraphy en el Nano (no funcional)

**GAP confirmado:** Polygraphy **NO funciona en JetPack 4.6.1** porque Python 3.6.9 es incompatible con polygraphy 0.45+ (requiere Py 3.8+). En el Nano, validar con `trtexec` directo (no polygraphy).

Referencia: [foro NVIDIA #349598](https://forums.developer.nvidia.com/t/how-to-generate-and-verify-an-int8-calibration-cache-cache-for-trtexec-on-on-jetson-nano-tensorrt-8-2-1-8-polygraphy-failing-on-device/349598) "How to generate and verify an INT8 calibration cache (.cache) for trtexec on Jetson Nano (TensorRT 8.2.1.8) — Polygraphy failing on-device".

### 5.8 ONNX Runtime + TRT EP en Nano (no viable)

ORT + TRT EP requiere CUDA 11.4 (ORT 1.11+); Nano tiene CUDA 10.2. **Ruta no viable.** Inferencia en el Nano usa TRT Python bindings directos + `cuda-python 11.0`.

---

## 6. Engine TRT compilado en el Nano (D8)

### 6.1 Regla vinculante

**El `.engine` siempre se compila en el Nano**, nunca en Vast.ai ni transferido. Justificación: TensorRT engines son **GPU-architecture-specific y TRT-version-specific**. Un engine compilado en RTX 4090 (Ada `sm_89`, TRT 10) NO ejecutará en Jetson Nano (Maxwell `sm_53`, TRT 8.2.1). Incluso con la misma versión de TRT, las fusiones de kernels difieren entre `sm_89` y `sm_53` por catálogo de tactics.

### 6.2 Comando build en el Nano

```bash
# En el Nano
trtexec --onnx=best.onnx \
        --fp16 \
        --workspace=1024 \
        --saveEngine=best.engine \
        --verbose 2>&1 | tee trt_build.log
```

**Quirk OOM (issue [`ultralytics/ultralytics#14751`](https://github.com/ultralytics/ultralytics/issues/14751)):** TRT build necesita workspace; el Nano tiene 4 GB compartida CPU/GPU. `--workspace=1024` MB es el sweet spot. Si OOM persiste, bajar a 512.

### 6.3 Track A no aplica

Track A corre en **CPU TFLite + XNNPACK + NEON**, no usa TensorRT. No hay paso de "compilar engine". El `.tflite` se carga directamente vía `tflite_runtime.Interpreter`.

---

## 7. INT8 en Maxwell `sm_53`: el cierre del gap (D14)

### 7.1 Resumen del problema

La literatura 2024–2026 presenta **evidencia directamente contradictoria** sobre INT8 PTQ en Jetson Nano B01 (Maxwell `sm_53`, TensorRT 8.2.1). No existe ningún paper peer-reviewed ni preprint arXiv que haya caracterizado el trade-off directamente sobre Maxwell `sm_53`. El gap declarado en Ronda 4 es **irreductible con la literatura actual**.

### 7.2 Mecanismo teórico decisivo: ausencia de `dp4a`

Maxwell `sm_53` **carece de la instrucción `dp4a` (dot product de 4 × INT8 acumulando a INT32)** introducida en Pascal `sm_61` (2016). Sin `dp4a`, TensorRT tiene tres opciones:

1. **Usar kernels CUDA INT8 SIMD vía `dp4a`** → **no disponible en `sm_53`**.
2. **Emular INT8 vía FP16/FP32** → elimina cualquier beneficio de velocidad y añade overhead.
3. **Mixed precision fallback** → TensorRT revierte la capa a FP16, generando grafo mixto con conversiones adicionales.

Confirmación oficial NVIDIA (issue [`NVIDIA/TensorRT#3762`](https://github.com/NVIDIA/TensorRT/issues/3762)):

> *"`--int8` means Enable int8 precision, in addition to fp32."*

Es decir, INT8 nunca reemplaza FP32: lo complementa, y las capas no cuantizables revierten.

**Conclusión mecánica:** el speedup INT8 en Maxwell `sm_53` es **estructuralmente nulo** por arquitectura de hardware. La degradación de mAP ocurre igualmente (cuantización modifica pesos y activaciones independientemente del hardware), pero sin la contraparte de velocidad que la justifique.

### 7.3 Evidencia primaria (fuentes contradictorias)

#### Fuente 1 — Qengineering (confianza media-alta, repo activo TRT 8.x para Nano B01)

[`Qengineering/YoloV8-TensorRT-Jetson_Nano`](https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano), rama `tensorrt8`:

> *"All models are quantized to `FP16`. The `int8` models don't give any increase in FPS, while, at the same time, their mAP is significantly worse."*

Tabla FP16 reportada:

| Modelo | Nano B01 (FPS) | Orin Nano (FPS) |
|--------|----------------|------------------|
| YOLOv8n | 19 | 100 |
| YOLOv8s | 9,25 | 100 |

El autor **no publica tabla INT8** porque concluye que el upside es nulo y el daño a mAP es significativo.

#### Fuente 2 — espstack.com (confianza baja, sin metodología verificable)

[`espstack.com/blogs/posts/yolov8-jetson-nano.html`](https://espstack.com/blogs/posts/yolov8-jetson-nano.html):

| Modelo | Formato | FPS | Latencia (ms) | mAP50 |
|--------|---------|-----|---------------|-------|
| YOLOv8n | PyTorch FP32 | 7–9 | 110–140 | 0,887 |
| YOLOv8n | TRT FP16 | 18–22 | 45–55 | 0,885 |
| YOLOv8n | TRT INT8 | 28–32 | 31–36 | 0,878 |

Esta fuente reporta caída de mAP50 FP16 vs INT8 de solo 0,7 pp y +50% FPS. Pero:

- (a) dataset de calibración no especificado,
- (b) mAP50 0,887 coherente con COCO no con custom 3 clases,
- (c) sin código reproducible,
- (d) **contradice directamente a Qengineering**.

#### Fuente 3 — `the0807/YOLOv8-ONNX-TensorRT` (Orin Nano `sm_87`, con TC INT8 reales)

| Cuantización | FPS | mAP val 50–95 |
|--------------|-----|---------------|
| FP16 | 60 | 37,1 |
| INT8 | 63 | 33,0 |

Drop mAP50-95: −4,1 pp con apenas +5% FPS. Sobre hardware **con** Tensor Cores INT8. Esto sugiere que incluso en arquitecturas modernas el speedup es modesto para nano/small.

### 7.4 Decisión D14 — FP16-only por default

**Track B se queda en FP16-only por default.** Experimento INT8 opcional 45–60 min en el propio Jetson Nano, **únicamente si hay margen de tiempo antes de la entrega**.

### 7.5 Protocolo del experimento INT8 opcional en Nano

```bash
# 1. Compilar engine INT8 con calibración
trtexec --onnx=yolov8n_custom.onnx \
        --saveEngine=yolov8n_int8.engine \
        --int8 \
        --calib=calib_list.txt \
        --workspace=1024

# Generar calib_list.txt con paths a imágenes de calibración (~100-500 imágenes del val set)

# 2. Medir mAP@0.5 en val set completo
python validate_engine.py --engine yolov8n_int8.engine --data data.yaml

# 3. Medir FPS empírico
trtexec --loadEngine=yolov8n_int8.engine --iterations=100
# Comparar contra FP16:
trtexec --loadEngine=yolov8n_fp16.engine --iterations=100
```

### 7.6 Criterio binario de decisión

Definido en D14 (ledger):

- **Si** `FPS_INT8 < FPS_FP16 × 1,10` (menos de 10% de ganancia en FPS) **O** `mAP_INT8 < mAP_FP16 − 5 pp`, **abandonar INT8** y consolidar FP16-only.
- **Si NO** (FPS gana ≥ 10% **Y** mAP cae < 5 pp), **adoptar INT8**.

**Importante:** zona gris (entre +0% y +10% FPS) → abandonar por el criterio del 10%. Evitamos optimización ambigua que añade complejidad sin gain claro.

### 7.7 Calibración INT8 — quirks documentados

- Foro NVIDIA #349598 confirma que **Polygraphy + INT8 falla en JetPack 4.6.1** (incompat. Py 3.6). Usar `IInt8EntropyCalibrator2` Python custom + `trtexec --int8 --calib=<cache>`.
- Foro NVIDIA #331356 confirma que `TRT INT8 conversion fails with assertion error using Ultralytics` en Orin → quirk de Ultralytics export, no del runtime. Mitigación: re-export ONNX manualmente y feed a trtexec directo.

---

## 8. Pipeline completo de validación pre-deploy

### 8.1 Track A

```bash
source /opt/venv/tracka/bin/activate

# Gate 1 — inspección op_version
pip install tflite==2.5.0
python scripts/validate_tflite_ops.py track_a/exports/model_int8.tflite
# Exit code != 0 si alguna op excede runtime 2.5

# Gate 2 — carga test
pip install "https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp38-cp38-linux_x86_64.whl"
python scripts/validate_tflite_load.py track_a/exports/model_int8.tflite

# Si Gate 2 falla con TFLite_Detection_PostProcess missing:
#   → Documentar para aplicar D15 (wheel Coral CP36) en el Nano
#   → NO abortar export, el Plan B existe
```

### 8.2 Track B

```bash
source /opt/venv/trackb/bin/activate

# Gate 3 — inspección ops contra blacklist TRT 8.2
pip install onnx
python scripts/validate_onnx_ops.py track_b/exports/best.onnx
# Exit code != 0 si hay ops blacklisted o ConstantOfShape no-FP32

# Gate 4 — Polygraphy en Docker NGC
docker pull nvcr.io/nvidia/tensorrt:21.11-py3
docker run --rm --gpus all -v "$(pwd)":/ws nvcr.io/nvidia/tensorrt:21.11-py3 \
  bash -c "pip install -q polygraphy onnx && \
           polygraphy run /ws/track_b/exports/best.onnx \
             --onnxrt --trt --atol 1e-2 --rtol 1e-2 \
             --input-shapes images:[1,3,416,416]"
# Exit code != 0 si TRT 8.2 no puede construir el engine o si los outputs divergen
```

### 8.3 Script `validate_artifacts.py` (#5' del HANDOFF)

CLI consolidado con flags `--track {A,B}` y `--model <path>`. Esquema:

```python
# scripts/validate_artifacts.py
"""
Validación pre-deploy unificada para artefactos embebidos-3.

Uso:
    python validate_artifacts.py --track A --model track_a/exports/model_int8.tflite
    python validate_artifacts.py --track B --model track_b/exports/best.onnx
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

def validate_track_a(model_path: Path) -> dict:
    """Gate 1 (op_version) + Gate 2 (load test) para Track A."""
    result = {"track": "A", "model": str(model_path), "gates": {}}

    # Gate 1
    # ... (código de §3.1)
    result["gates"]["op_version"] = {"status": "pass", "ops_inspected": N, "violations": []}

    # Gate 2
    # ... (código de §3.2)
    result["gates"]["load_test"] = {"status": "pass", "outputs": 4, "input_dtype": "uint8"}
    result["gates"]["tflite_detection_postprocess"] = {"status": "pass"}  # o "fallback_d15_required"

    return result

def validate_track_b(model_path: Path) -> dict:
    """Gate 3 (ops blacklist) + Gate 4 (polygraphy en Docker NGC) para Track B."""
    result = {"track": "B", "model": str(model_path), "gates": {}}

    # Gate 3
    # ... (código de §4.4)
    result["gates"]["ops_blacklist"] = {"status": "pass", "ops_present": [...], "violations": []}

    # Gate 4
    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{model_path.parent.parent}:/ws",
        "nvcr.io/nvidia/tensorrt:21.11-py3",
        "bash", "-c",
        f"pip install -q polygraphy onnx && "
        f"polygraphy run /ws/{model_path.name} --onnxrt --trt "
        f"--atol 1e-2 --rtol 1e-2 --input-shapes images:[1,3,416,416]"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    result["gates"]["polygraphy"] = {
        "status": "pass" if proc.returncode == 0 else "fail",
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
    }

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["A", "B"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("validation_report.json"))
    args = parser.parse_args()

    if args.track == "A":
        report = validate_track_a(args.model)
    else:
        report = validate_track_b(args.model)

    # Persistir JSON + log markdown
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # Exit code != 0 si algún gate falló
    all_pass = all(g.get("status") == "pass" for g in report["gates"].values())
    sys.exit(0 if all_pass else 1)

if __name__ == "__main__":
    main()
```

**Implementación completa pendiente en tarea #5' del HANDOFF.**

---

## 9. Riesgos residuales y mitigaciones

| # | Riesgo / GAP | Mitigación |
|---|--------------|------------|
| R1 | TF 2.15 puede generar `op_version > 2.5 max` para `DEPTHWISE_CONV_2D` o `FULLY_CONNECTED` sin documentación pública | Inspección obligatoria del flatbuffer (Gate 1). Workaround `flatbuffer_utils.py` si falla. Re-train con converter legacy. |
| R2 | Wheel NVIDIA `tensorflow==2.5.0+nv21.8` puede no incluir `TFLite_Detection_PostProcess` (sin confirmación verbatim) | Fallback Coral wheel CP36 aarch64 (D15). Verificación anticipada en Gate 2. |
| R3 | Drop INT8 YOLOv8n Maxwell `sm_53` no caracterizado en literatura | Confirmado FP16-only por mecanismo (`dp4a` ausente). Experimento opcional 1 h en Nano (D14). |
| R4 | Polygraphy en container Vast.ai (CUDA 12.4) instala TRT 10 por defecto, no 8.2 | Docker NGC `tensorrt:21.11-py3` obligatorio (Gate 4). |
| R5 | Validación x86 con TRT 8.2.1 no equivale a Maxwell `sm_53` (fusiones de kernels difieren) | Gate necesario pero no suficiente. Smoke test rápido en Nano (5–15 min) antes del primer training completo. |
| R6 | EfficientNMS_TRT plugin roto en Maxwell con TRT 8.x (issue #1538) | `nms=False` en export ONNX; NMS en CPU NumPy con `cv2.dnn.NMSBoxes` en Nano. |
| R7 | TF 2.15 INT8 PTQ puede no preservar `TFLite_Detection_PostProcess` si el SavedModel se reconstruye | Validar pre-export con `tf.saved_model.load()` y revisar signatures. Gate 2 (carga test) confirma post-export. |
| R8 | `ConstantOfShape` con tipos no-FP32 en ONNX por upgrades futuros de Ultralytics | Inspección Gate 3 detecta y bloquea. Re-export con flags conservadores si aparece. |
| R9 | Docker daemon no disponible en Vast.ai (Docker-in-Docker) | Verificación: `docker info` en bootstrap. Si falla, ejecutar Gate 4 en máquina x86 separada (local Windows con Docker Desktop). |
| R10 | Build TRT engine en Nano falla por OOM (issue #14751) | `trtexec --workspace=1024` (vs default que puede ser mayor); cerrar JupyterLab/desktop antes de build. |

---

## 10. Fuentes consultadas

| # | Título | URL | Tipo |
|---|--------|-----|------|
| 1 | TFLite CONV_2D version 5 (issue #41943) | https://github.com/tensorflow/tensorflow/issues/41943 | Issue |
| 2 | TFLite quantized CONV_2D v5 (issue #50652) | https://github.com/tensorflow/tensorflow/issues/50652 | Issue |
| 3 | TFLite MobileNet v2 CONV_2D v5 RPi (issue #43232) | https://github.com/tensorflow/tensorflow/issues/43232 | Issue |
| 4 | TFLite FULLY_CONNECTED v12 Android (issue #80736) | https://github.com/tensorflow/tensorflow/issues/80736 | Issue |
| 5 | TFLite forward-compat schema (issue #62413) | https://github.com/tensorflow/tensorflow/issues/62413 | Issue |
| 6 | Tflite_micro flatbuffers aarch64 (issue #2703) | https://github.com/tensorflow/tflite-micro/issues/2703 | Issue |
| 7 | Google Coral tflite-runtime wheels | https://google-coral.github.io/py-repo/tflite-runtime/ | Doc oficial |
| 8 | PINTO0309/Tensorflow-bin | https://github.com/PINTO0309/Tensorflow-bin | Repo |
| 9 | Qengineering/TensorFlow-JetsonNano (wheels) | https://github.com/Qengineering/TensorFlow-JetsonNano | Repo |
| 10 | Qengineering/TensorFlow_Lite_SSD_Jetson-Nano | https://github.com/Qengineering/TensorFlow_Lite_SSD_Jetson-Nano | Repo |
| 11 | tflite-support metadata_schema.fbs | https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/metadata/metadata_schema.fbs | Código fuente |
| 12 | TensorFlow version compatibility docs | https://www.tensorflow.org/guide/versions | Doc oficial |
| 13 | flatbuffer_utils.py (HEAD) | https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/flatbuffer_utils.py | Código fuente |
| 14 | onnx-tensorrt operators.md release/8.2-GA | https://github.com/onnx/onnx-tensorrt/blob/release/8.2-GA/docs/operators.md | Doc oficial |
| 15 | Polygraphy CHANGELOG | https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/CHANGELOG.md | Doc oficial |
| 16 | Polygraphy debug_accuracy | https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/how-to/debug_accuracy.md | Doc oficial |
| 17 | Polygraphy work_with_reduced_precision | https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/how-to/work_with_reduced_precision.md | Doc oficial |
| 18 | NGC TensorRT containers | https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt | Doc oficial |
| 19 | NVIDIA TensorRT 8.2.2 Support Matrix | https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-822/support-matrix/index.html | Doc oficial |
| 20 | NVIDIA TensorRT 10.0 blog (parser errors) | https://developer.nvidia.com/blog/nvidia-tensorrt-10-0-upgrades-usability-performance-and-ai-model-support/ | Blog oficial |
| 21 | trtexec benchmarking docs | https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html | Doc oficial |
| 22 | NVIDIA/TensorRT issue #1538 (EfficientNMS Maxwell) | https://github.com/NVIDIA/TensorRT/issues/1538 | Issue |
| 23 | NVIDIA/TensorRT issue #1994 (SSD MV2 FPNLite Nano) | https://github.com/NVIDIA/TensorRT/issues/1994 | Issue |
| 24 | NVIDIA/TensorRT issue #3732 (Polygraphy + Jetson) | https://github.com/NVIDIA/TensorRT/issues/3732 | Issue |
| 25 | NVIDIA/TensorRT issue #3762 (INT8 slower) | https://github.com/NVIDIA/TensorRT/issues/3762 | Issue |
| 26 | NVIDIA/TensorRT issue #4383 (Gather rank-0 opset 19) | https://github.com/NVIDIA/TensorRT/issues/4383 | Issue |
| 27 | NVIDIA forum #349598 (Polygraphy + INT8 Nano) | https://forums.developer.nvidia.com/t/how-to-generate-and-verify-an-int8-calibration-cache-cache-for-trtexec-on-on-jetson-nano-tensorrt-8-2-1-8-polygraphy-failing-on-device/349598 | Foro oficial |
| 28 | NVIDIA forum #331356 (TRT INT8 Ultralytics Orin) | https://forums.developer.nvidia.com/t/tensorrt-int8-conversion-fails-with-assertion-error-using-ultralytics/331356 | Foro oficial |
| 29 | NVIDIA forum 299957 (TRT 8.2 MMU fault) | https://forums.developer.nvidia.com/t/tensorrt-8-2-triggers-mmu-fault-on-a-specific-onnx-model/299957 | Foro oficial |
| 30 | Qengineering YoloV8-TensorRT-Jetson_Nano | https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano | Repo |
| 31 | espstack.com YOLOv8 on Jetson Nano | https://espstack.com/blogs/posts/yolov8-jetson-nano.html | Blog |
| 32 | the0807/YOLOv8-ONNX-TensorRT | https://github.com/the0807/YOLOv8-ONNX-TensorRT | Repo |
| 33 | triple-Mu/YOLOv8-TensorRT | https://github.com/triple-mu/YOLOv8-TensorRT | Repo |
| 34 | Linaom1214/TensorRT-For-YOLO-Series issue #112 | https://github.com/Linaom1214/TensorRT-For-YOLO-Series/issues/112 | Issue |
| 35 | Alqahtani et al. 2024 — Benchmarking edge devices | https://arxiv.org/abs/2409.16808 | Paper arXiv |
| 36 | NobuoTsukamoto/tensorrt-examples | https://github.com/NobuoTsukamoto/tensorrt-examples/blob/main/python/detection/README.md | Repo |
| 37 | ONNX Runtime TRT EP requirements | https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html | Doc oficial |
| 38 | Ultralytics issue #14751 (TRT engine OOM Nano) | https://github.com/ultralytics/ultralytics/issues/14751 | Issue |
| 39 | Ultralytics issue #2821 (TRT INT64 weights) | https://github.com/ultralytics/ultralytics/issues/2821 | Issue |
| 40 | foro NVIDIA TRT 8.2 on Jetson 193833 | https://forums.developer.nvidia.com/t/tensorrt-8-2-on-jetson/193833 | Foro oficial |
| 41 | foro NVIDIA Polygraphy on Nano 225635 | https://forums.developer.nvidia.com/t/can-jetson-nano-use-polygraphy-in-tensorrt-and-how-do-i-install-it/225635 | Foro oficial |
| 42 | foro NVIDIA Jetson Nano DL benchmarking review | https://www.themoonlight.io/en/review/benchmarking-deep-learning-models-on-nvidia-jetson-nano-for-real-time-systems-an-empirical-investigation | Review |

---

## 11. Cross-references

- **[`decisiones-D1-D15-ledger.md`](decisiones-D1-D15-ledger.md)** — D8, D12, D13, D14, D15 detallados.
- **[`compatibilidad-stack-cloud-jetson.md`](compatibilidad-stack-cloud-jetson.md)** — Tabla `op_version` §6.6 (input para Gate 1); ops opset 11 §7.3 (input para Gate 3); mecanismo `dp4a` ausente §2.3 (input para D14).
- **[`infraestructura-training-vastai-uv-hf.md`](infraestructura-training-vastai-uv-hf.md)** — Bootstrap instala Docker para Gate 4; `validate_artifacts.py` se ejecuta antes del auto-destroy.
- **[`dataset-roboflow-yolov8.md`](dataset-roboflow-yolov8.md)** — Flags de export ONNX que producen el `.onnx` validado por Gates 3 y 4.
- **[`HANDOFF-implementacion-vastai-hf.md`](HANDOFF-implementacion-vastai-hf.md)** — Tarea #5' (`validate_artifacts.py`) usa el esquema de §8.3.

---

**Fin del documento.** Cualquier cambio a D8, D12, D13, D14 o D15 requiere nueva ronda `/investiga`.
