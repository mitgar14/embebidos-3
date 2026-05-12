# Validación rápida de cámara USB en Windows (pre-compra)

> **Propósito:** confirmar en ≤ 7 minutos que la unidad Logitech (C920 OG o C930e) está en buen estado antes de pagar al vendedor.
> **Contexto del proyecto:** MVP clasificador de residuos `embebidos-3`. El despliegue final es Jetson Nano + L4T R32.7.x, pero las primeras pruebas se hacen sobre Windows para no demorar al vendedor.
> **Stack objetivo en producción:** GStreamer + `nvv4l2decoder mjpeg=1` (fallback `nvjpegdec`) + TensorRT YOLOv8n FP16 @ 416×416.
> **Fecha:** 2026-05-11.

---

## 0. ¿Necesito instalar software específico para C920 o C930e?

**Respuesta corta: NO para hacer la prueba básica.**

Logitech confirma oficialmente que sus webcams son **UVC class-compliant** (USB Video Class). El driver lo provee el propio Windows:

| Fuente oficial | Cita textual |
|----------------|--------------|
| [C920-C Technical Specifications](https://support.logi.com/hc/en-us/articles/360023307774) | *"Software Support (at release): No software drivers to install. Webcam uses native UVC drivers."* |
| [C930e Technical Specifications](https://support.logi.com/hc/en-us/articles/360023302054) | *"Software Support: No software required. Logitech Utility Software available to control pan, tilt and zoom from your computer."* |
| [Capture requirements](https://support.logi.com/hc/en-us/articles/360060226873) | *"You do not need Capture installed in order to use your webcam. Our webcams are USB Video Class (UVC) devices and driver support is built into Microsoft Windows 10 and macOS."* |

### Software opcional de Logitech — solo si quieres control fino

| Software | Para qué sirve | Estado 2026 | Compatible C920 OG | Compatible C930e |
|----------|----------------|-------------|----------------------|---------------------|
| **Logitech G HUB** | Ajustar exposure, white balance, focus, zoom, filtros, perfiles. Pensado para gaming/streaming. | **Activo y recomendado oficial** | Sí | Parcial (preferir Logi Tune) |
| **Logi Tune (Desktop)** | Control para webcams business: pan/tilt/zoom, color presets, manual focus, firmware updates. | **Activo y recomendado oficial para business** | No listado oficialmente | **Sí (es la app oficial)** |
| **Logitech Capture** | Grabación local con overlays, transiciones, ChromaKey. | **Descontinuado en 2022** (versión final, sin updates futuros). Logitech dice: *"download Logitech G HUB if you want to adjust compatible webcam settings"* | Sí (legacy) | Sí (legacy) |
| **Logi Options+** | Configuración general para mice/teclados/webcams modernos. | Activo, pero limitado para webcams viejas. | No (no aparece en la lista oficial) | No (no aparece) |
| **Logitech Webcam Software 2.80 (LWS)** | App histórica para C170/C270/C310/C525/C615/C920. Última versión 2012-10-27. | **Legacy total** | Sí (legacy) | No |
| **Logitech Gaming Software (LGS) 8.85+** | Predecesor de G HUB. | Descontinuado | Sí (legacy) | No |

### Recomendación para la prueba pre-compra

| Caso | Qué instalar antes de salir |
|------|---------------------------------|
| **Mínimo viable** (test básico de funcionamiento) | NADA. Windows 10/11 reconoce la cámara via UVC nativo. La app "Cámara" del sistema basta. |
| **Recomendado** (verificar MJPG@720p — gate crítico para Jetson) | `ffmpeg` (vía `winget install Gyan.FFmpeg`) + `OBS Studio` (`winget install OBSProject.OBSStudio`). |
| **Opcional** (probar ajustes finos solo si el vendedor permite) | Para C920 OG: G HUB. Para C930e: Logi Tune. Pero no son necesarios para validar la unidad. |

> **Conclusión operativa:** el día de la compra solo necesitas Windows actualizado, `ffmpeg` y OBS. NO descargues ni instales G HUB / Logi Tune en el sitio — desperdicia tiempo del vendedor y no aporta a la decisión de compra.

---

## 1. Preparación antes de salir (5 min en casa)

```powershell
winget install OBSProject.OBSStudio
winget install Gyan.FFmpeg
```

Después de instalar `ffmpeg`, **cerrar y reabrir PowerShell** para que el PATH se refresque. Verifica con:

```powershell
ffmpeg -version
```

Checklist final antes de salir:

- [ ] `ffmpeg` instalado y respondiendo.
- [ ] OBS Studio instalado, scene "Test" creada con un *Video Capture Device* placeholder.
- [ ] App "Cámara" de Windows pre-abierta (Win+S → "Cámara").
- [ ] PowerShell abierto en una ventana.
- [ ] Discord / Zoom / Teams / Skype / navegador con cámara activa → CERRADOS (bloquean el dispositivo).
- [ ] Puerto USB-A libre y directo del laptop (no hub, no dock, no extensor).
- [ ] Linterna pequeña o luz del celular (para verificar lente sin reflejos).

---

## 2. En el sitio — 5 pasos cronometrados (≈ 6 minutos)

### Paso 1 (30 s) — Inspección física

| Item | Cómo se verifica | Rojo (rechazar) |
|------|-------------------|------------------|
| Cable USB integrado | Sin cortes ni alma de cable visible, conector USB-A sin pines doblados ni oxidados. | Cortes / pines doblados / oxidación. |
| Lente | Sin rayones profundos, sin hongos (manchas blanco-lechosas internas), sin polvo grueso interno. Iluminar con linterna en ángulo. | Hongos visibles / rayones que crucen el centro del lente. |
| Clip de montaje | Cierra con tensión, articulación firme. | Grietas en la articulación / clip flojo. |
| Sticker inferior | Visible y legible. Anota M/N y serie. | Sticker arrancado o ilegible (sin garantía si hay claim a Logitech). |
| Logo + branding | Foto al lente para anotar la revisión. | – |

**Identificación rápida de modelo/revisión por marcadores visuales:**

| Indicio visual | Modelo/revisión probable |
|----------------|---------------------------|
| Logo "ojo verde" antiguo + texto "Carl Zeiss" sobre el lente + M/N **V-U0028** | **C920 Rev 1 OG** (la valiosa, con encoder H.264 hardware) |
| Logo viejo o "Logi" wordmark + sin Carl Zeiss + M/N **V-U0060** | **C920 Rev 2 OrbiCam** (sin H.264 hardware) |
| Logo "Logi" moderno + M/N **V-U0068** / **V-U0070** | **C920 Rev 3 / C920e / C920s** |
| Anillo de lente **cromado/plateado** + branding "Logitech" sobre el cristal + M/N **V-U0029** | **C930e** (NO es C920) |

### Paso 2 (1 min) — Conexión + reconocimiento Windows

Enchufa al USB del laptop. Espera 5–10 s. En el PowerShell ya abierto:

```powershell
Get-PnpDevice -PresentOnly |
  Where-Object { $_.FriendlyName -match "Logitech|C920|C922|C930|HD Pro Webcam" } |
  Format-Table FriendlyName, InstanceId, Status -AutoSize
```

**Resultado esperado:**

- `Status = OK` (no `Error`, no `Unknown`, no `Degraded`).
- `InstanceId` contiene uno de estos hardware IDs:

| Hardware ID | Modelo/revisión |
|-------------|-------------------|
| `USB\VID_046D&PID_082D` | C920 OG (Rev 1) |
| `USB\VID_046D&PID_0892` | C920 Rev 2 / Rev 3 (heredado) |
| `USB\VID_046D&PID_08E5` | C920 Rev 3 (algunas unidades) |
| `USB\VID_046D&PID_0843` | C930e |
| `USB\VID_046D&PID_0825` | C270 |

**Si el `Status` no es `OK` o no aparece el dispositivo → rechaza la unidad.**

### Paso 3 (2 min) — Test funcional en app Cámara

Abre la app **Cámara**. Debe ver preview en vivo en menos de 2 s.

| Verificación | Cómo hacerla | OK | Rojo |
|--------------|----------------|-----|------|
| Sensor sano | Mirar preview a pantalla completa por 10 s. | Sin franjas verdes/rosas, sin puntos fijos, sin bandeo horizontal. | Cualquier artefacto persistente. |
| Autofocus | Pasar la mano de 1 m a 15 cm de la cámara. | Enfoca progresivamente en <1 s, estabiliza. | "Hunting" continuo / nunca enfoca cerca. |
| LED indicador | Mirar la cámara mientras hay preview. | LED blanco encendido. | LED apagado o intermitente. |
| Micrófono | Tocar dos veces el costado de la cámara, luego decir "uno, dos, tres". | En la app sube el indicador de audio (o graba 5 s y reproduce con sonido claro). | Sin captura de audio. |
| Cambio de resolución | Configuración → 1080p. | Preview sigue fluido (≥ 25 fps subjetivos). | Lag visible / frames negros. |
| Foto + video | Capturar foto + video de 10 s a 1080p. Reproducir el video. | Sin frames negros, sin desincronía audio/video. | Cualquier defecto grave. |

### Paso 4 (1 min) — Verificación MJPG @ 720p (CRÍTICO para Jetson)

En PowerShell. Primero lista los dispositivos disponibles:

```powershell
ffmpeg -hide_banner -list_devices true -f dshow -i dummy 2>&1 | Select-String -Pattern "video"
```

Anota el nombre exacto de la cámara (por ejemplo `HD Pro Webcam C920` o `Logitech Webcam C930e`). Luego:

```powershell
$cam = "HD Pro Webcam C920"   # ← reemplazar con el nombre exacto del comando anterior
ffmpeg -hide_banner -f dshow -list_options true -i video="$cam" 2>&1 |
  Select-String -Pattern "mjpeg|yuyv422|h264"
```

**Líneas que DEBEN aparecer:**

```
[dshow ...] pixel_format=yuyv422  min s=...  max s=1920x1080 fps=...
[dshow ...] vcodec=mjpeg          min s=...  max s=1920x1080 fps=...
[dshow ...] vcodec=mjpeg          min s=...  max s=1280x720  fps=30
```

**Bonus C920 Rev 1 OG** (no es eliminatorio, pero confirma la revisión "valiosa"):

```
[dshow ...] vcodec=h264   ...
```

Si **NO aparece** `vcodec=mjpeg` con `1280x720` y `fps=30` → la cámara no sirve para el pipeline del Jetson aunque luzca bien en preview. **Rechazar la unidad.**

### Paso 5 (1 min) — Captura de evidencia en OBS

En OBS abre la scene "Test", añade *Video Capture Device*:

1. **Device:** selecciona la cam por nombre.
2. **Resolution/FPS Type:** `Custom`.
3. **Resolution:** `1280x720`.
4. **FPS:** `30`.
5. **Video Format:** `MJPEG`.

Si OBS muestra preview estable durante **30 s a 30 fps sin tearing ni dropped frames** (panel inferior derecho debería marcar `30.00 fps` y `0 lagged / 0 skipped`) → ✅ **unidad apta**.

---

## 3. Resumen ejecutivo (qué NO saltarse, en orden)

1. **Sticker M/N visible + estado físico** — 30 s, sin tocar el laptop.
2. **`Get-PnpDevice` muestra Status OK + VID/PID Logitech esperado** — 1 min.
3. **Preview en app Cámara + autofocus + LED + mic** — 2 min.
4. **`ffmpeg -list_options` confirma MJPG 1280×720@30** — 1 min. **Gate eliminatorio para el proyecto Jetson, no se salta.**
5. **OBS captura 30 s estables a 30 fps en MJPEG** — 1 min.

**Tiempo total ≈ 6 min.**

Reglas de decisión:

- Pasos **1, 2 y 4 fallidos** → rechazar la unidad sin discusión.
- Pasos **3 y 5 con defectos menores** → tolerable si los pasos eliminatorios pasan, negociar precio.

---

## 4. Apéndice — Comando único para Paso 2 + 4 (ahorra tiempo)

Guarda este script en `scripts/validate-webcam.ps1` y ejecuta `.\validate-webcam.ps1` el día de la prueba:

```powershell
# scripts/validate-webcam.ps1
Write-Host "=== Logitech webcam validation ===" -ForegroundColor Cyan

Write-Host "`n[1/3] Dispositivo Plug-and-Play:" -ForegroundColor Yellow
Get-PnpDevice -PresentOnly |
  Where-Object { $_.FriendlyName -match "Logitech|C920|C922|C930|HD Pro Webcam" } |
  Format-Table FriendlyName, InstanceId, Status -AutoSize

Write-Host "`n[2/3] Dispositivos de video DirectShow:" -ForegroundColor Yellow
$dshowOutput = ffmpeg -hide_banner -list_devices true -f dshow -i dummy 2>&1
$dshowOutput | Select-String -Pattern '"(.*?)"\s*\(video\)' | ForEach-Object { $_.Matches.Groups[1].Value }

Write-Host "`n[3/3] Formatos soportados (MJPG/H264/YUYV):" -ForegroundColor Yellow
$camName = ($dshowOutput | Select-String -Pattern '"(.*?Logi.*?)"\s*\(video\)' | Select-Object -First 1).Matches.Groups[1].Value
if ($camName) {
  Write-Host "Camara detectada: $camName" -ForegroundColor Green
  ffmpeg -hide_banner -f dshow -list_options true -i video="$camName" 2>&1 |
    Select-String -Pattern "mjpeg|yuyv422|h264|1280x720|1920x1080"
} else {
  Write-Host "No se detecto camara Logitech via DirectShow." -ForegroundColor Red
}
```

---

## 5. Tras la compra — siguiente paso (no aplica el día de la prueba)

Una vez en posesión de la cámara, mover la validación a Jetson Nano y ejecutar:

```bash
# En el Jetson Nano (JetPack 4.6.x):
v4l2-ctl --list-devices
v4l2-ctl --list-formats-ext -d /dev/video0    # confirmar MJPG 1280x720 @ 30 fps
lsusb -v -d 046d:082d | grep -E "bcdDevice|iProduct"   # ajustar PID según modelo
```

Después correr el pipeline canónico de la sección 5.3 de
`investigaciones/2026-05-10/2026-05-10-camara-usb-jetson-nano.md`
con `fpsdisplaysink` durante 60 s antes de integrar al harness `scripts/bench_jetson.py`.
