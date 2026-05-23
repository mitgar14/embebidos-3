# Tailscale bloqueado en red UAO — diagnóstico y workarounds

**Dominio:** acceso-remoto / tailscale / redes-corporativas
**Foco:** Tailscale (mesh VPN sobre WireGuard) degradado o bloqueado cuando el AP TP-Link de la Jetson Nano del proyecto embebidos-3 está conectado a la WiFi de la Universidad Autónoma de Occidente (UAO, Cali, Colombia).

---

## Resumen ejecutivo

El SSH al Jetson Nano del proyecto embebidos-3 va por **Tailscale**, no por SSH directo capa-3. Tailscale funciona en dos planos: un **control plane** (HTTPS a `controlplane.tailscale.com`, autenticación + intercambio de claves WireGuard) y un **data plane** (UDP 41641 directo entre peers, o fallback DERP sobre TCP 443). Cualquier red corporativa que filtre cualquiera de los dos planos rompe la conectividad. La evidencia recolectada apunta a **tres mecanismos típicos de bloqueo en redes universitarias**, con probabilidad alta para el caso UAO:

1. **DNS filtering** de `*.tailscale.com` (más fácil de implementar para IT, primera línea de defensa).
2. **Application Control / categorización VPN** en firewalls Fortinet, Palo Alto, SonicWall, Cisco — detectan el handshake WireGuard por su firma de 148 bytes y lo bloquean en el primer paquete (issues GitHub `tailscale/tailscale#11789, #11791, #15217` confirman explícitamente el patrón con Fortinet/FortiGuard).
3. **UDP outbound deny-all** excepto DNS/NTP — cae todo el tráfico WireGuard directo; Tailscale debería fallback a DERP TCP 443, pero si simultáneamente se filtra `*.tailscale.com` por SNI/DPI, también falla DERP.

**Diagnóstico in-situ requiere ejecutar `tailscale netcheck` desde la UAO** — el comando reporta en 3-5 segundos cuál de los tres vectores aplica (`UDP: false`, `Nearest DERP: unknown`, `CaptivePortal: true/false`). Sin ese reporte, todas las hipótesis son especulativas.

**Workaround prioritario para sustentación universitaria (decisión técnica):**
- **Si solo UDP está bloqueado y DERP TCP 443 funciona**: Tailscale ya hace fallback automático. Aceptar la latencia DERP (~88ms a Miami medidos) y proceder. Para SSH es perfectamente usable.
- **Si DERP también está bloqueado**: desplegar un **DERP propio en VPS** con dominio arbitrario y SNI passthrough (~5 USD/mes, configuración documentada).
- **Plan B garantizado**: **hotspot 4G** durante la sustentación (consenso de comunidad Level1Techs + r/Tailscale).

---

## Contexto del problema

### Topología actual verificada (2026-05-22, red residencial)

```
[Laptop NATASHA-5G — Windows]                     [Jetson Nano — Ubuntu 18.04 ARM64]
  IP local: 192.168.80.17                          IP local: 192.168.1.9
  Tailnet:  100.94.26.128  ←——Tailscale——→        Tailnet:  100.100.166.120
                                                   IP pública NAT: 186.169.20.111

Tailscale netcheck del Nano (red Nicolás):
  UDP: true
  IPv4: yes, 186.169.20.111:55720
  MappingVariesByDestIP: false  (NAT cone-friendly)
  CaptivePortal: false
  Nearest DERP: Miami (88.2 ms)
  Path activo: peer-to-peer DIRECT (latencia 45 ms)
```

### Síntoma reportado

SSH al Nano deja de funcionar cuando el AP TP-Link (que actúa como cliente WiFi en modo bridge/WISP/repetidor) se conecta a la WiFi de UAO en vez de a la red residencial de Nicolás (gateway 192.168.1.1).

### Restricciones del Nano

- **JetPack 4.6.1** (L4T R32.7.1, Ubuntu 18.04 ARM64).
- **Python 3.6.9**, TRT 8.2.1.8, sm_53 Maxwell.
- Cualquier workaround debe ser compatible con Ubuntu 18.04 ARM64. Sin Docker preferido (overhead). systemd v237 (sin features v240+).

---

## Track A — Hallazgos de agentes especializados

### A.1 Web (foros, blogs, comunidad)

**Hilos Reddit r/Tailscale más relevantes (todos confirman el patrón exacto del caso UAO):**

1. **"School Blocking Tailscale" (Fortinet)** — abril 2025
   `https://www.reddit.com/r/Tailscale/comments/1kbqx5c/`
   Estudiante con FortiGate institucional; Tailscale bloqueado explícitamente. Sin solución documentada en el hilo, pero confirma patrón.

2. **"Fortinet blocking Tailscale/WireGuard?"** — diciembre 2025
   `https://www.reddit.com/r/Tailscale/comments/1pm275j/`
   Hallazgo crítico: cambiar a `--port=443` UDP **NO ayuda** porque "Tailscale is only UDP. The only TCP part is the DERP." Si el firewall bloquea UDP en general, da igual el puerto.

3. **"Getting around blocks (SNI blocking DERP)"** — octubre 2024
   `https://www.reddit.com/r/Tailscale/comments/1g475fw/`
   Escuela bloquea DERP por SNI + todo UDP externo. Confirma que el bloqueo de UAO podría ser doble (UDP + SNI), no solo uno.

4. **"Can a self-hosted DERP relay circumvent blocking?"** — diciembre 2024
   `https://www.reddit.com/r/Tailscale/comments/1ha5wgs/`
   Diagnóstico clave: si `login.tailscale.com` (control plane) está bloqueado por DNS, **un DERP propio no soluciona el problema** porque Tailscale no puede ni autenticarse. Hay que verificar primero qué capa falla.

5. **"If WireGuard is blocked on my work network, will Tailscale?"** — febrero 2024
   `https://www.reddit.com/r/Tailscale/comments/1av3guf/`
   Respuesta validada: "Tailscale doesn't even need UDP ports to be unblocked, it'll use TCP over relay servers if a direct connection isn't possible." Confirma comportamiento esperado de DERP.

6. **"No more DERP relays on our university network"** — febrero 2025
   `https://www.reddit.com/r/Tailscale/comments/1iwhdnu/`
   Admin de red universitaria documentando la solución: editar `/etc/default/tailscaled` con `PORT="41642"` (alternativo) + port forwarding en el router.

**Level1Techs — "So... How would you bypass a firewall?"** — abril 2025
`https://forum.level1techs.com/t/so-how-would-you-bypass-a-firewall/229327`
Estudiante con ThreadRipper en universidad bloqueada. Bloquea `tailscale.com`, mirrors de Fedora y todo UDP no reconocido. Workarounds probados: ProtonVPN+443 (falla), ZeroTier (falla), WireGuard (falla). **Consenso final: router 4G/LTE independiente como solución más confiable cuando excepciones formales tardan semanas.** Advertencia: firewalls universitarios con subsidio ISP pueden tener dos capas (local + ISP).

**Hacker News — discusión sobre DPI y WireGuard fingerprinting**
`https://news.ycombinator.com/item?id=32199468`
Mecanismo técnico: WireGuard tiene un handshake de 148 bytes fijos (byte 0x01 + tres ceros + sender index 4B + ephemeral key 32B sin cifrar). Cualquier DPI con análisis de tamaño + primeros 4 bytes detecta el 100% de conexiones WireGuard en el primer paquete. Rusia (TSPU) bloquea WireGuard así con casi 100% de precisión.

**Blogs técnicos:**

- **Tailscale oficial — NAT traversal pt.1**: `https://tailscale.com/blog/nat-traversal-improvements-pt-1`
  Confirma que en redes con "firewall policy that outright blocks UDP", **0%** de conexiones directas, todo va por DERP.

- **fexyn.com — DPI Explained**: WireGuard "detection rate: ~100%, time to detect: first packet, method: 148-byte handshake fingerprint".

- **initez.nl — Understanding Tailscale**: documenta los tres vectores estándar que un admin universitario usaría: DPI signatures, firewall UDP rules, DNS filtering `controlplane.tailscale.com`. El tercero es el más barato y frecuente.

### A.2 Code (GitHub issues + alternativas)

**Issues `tailscale/tailscale` directamente aplicables al caso UAO:**

| # | Issue | Estado | Relevancia |
|---|---|---|---|
| [#11789](https://github.com/tailscale/tailscale/issues/11789) | OPEN | **"Tailscale ha sido bloqueado por fortiguard en categoría remote access"** — caso idéntico. Workaround: exit node externo via hotspot |
| [#11791](https://github.com/tailscale/tailscale/issues/11791) | OPEN | DERP maps NO se almacenan localmente. Si `login.tailscale.com/derpmap/default` está bloqueado, Tailscale no inicializa |
| [#13810](https://github.com/tailscale/tailscale/issues/13810) | OPEN | Eduroam: Tailscale DNS toma rol de default route, red lo bloquea. Workaround: `tailscale up --accept-dns=false` |
| [#15217](https://github.com/tailscale/tailscale/issues/15217) | OPEN | Issue específico DERP + Fortinet en múltiples sitios corporativos |
| [#4377](https://github.com/tailscale/tailscale/issues/4377) | OPEN | HTTPS DERP latency checks no funcionan con proxy SSL MITM |
| [#9524](https://github.com/tailscale/tailscale/issues/9524) | OPEN | Tailscale no puede sortear un DERP node bloqueado (incluso con DERP map custom) |
| [#19259](https://github.com/tailscale/tailscale/issues/19259) | OPEN | Regresión en v1.96.4: `UDP: false` falsamente reportado. Workaround: downgrade a 1.94.2 |
| [#1634](https://github.com/tailscale/tailscale/issues/1634) | OPEN | FR captive portal — resuelto parcialmente por [PR #12707](https://github.com/tailscale/tailscale/pull/12707) (julio 2024) |
| [#13119](https://github.com/tailscale/tailscale/issues/13119) | OPEN | FR: pluggable WireGuard obfuscation (AmneziaWG). Comentarios mencionan explícitamente Fortinet + SNI blocking de control server |
| [PR #16517](https://github.com/tailscale/tailscale/pull/16517) | MERGED jul-2025 | **`NodeAttrOnlyTCP443`**: cuando activo, deshabilita UDP+STUN+portmapper+peer-relay → todo por DERP TCP 443. Requiere activación desde control plane (Admin Console o headscale) |
| [PR #12161](https://github.com/tailscale/tailscale/pull/12161) | MERGED | Workaround para Palo Alto DIPP NAT misbehavior |

**Comandos canónicos de diagnóstico:**

```bash
# Estado general
tailscale status                          # peers + direct vs relay
tailscale status --json | jq '.Self,.Peer'
tailscale status --web --browser          # UI web (puerto 5252)

# Netcheck — comando principal de diagnóstico
tailscale netcheck                        # reporte estándar
tailscale netcheck --verbose              # logs internos de cada sonda STUN
tailscale netcheck --every=5s             # detectar intermitencia o captive portal tardío
tailscale netcheck --format=json          # output estructurado

# Ping con info de path
tailscale ping <peer>                     # latencia + tipo de path
tailscale ping --verbose <peer>           # detalle de paths intentados
tailscale ping --tsmp <peer>              # ping a nivel Tailscale Meta Protocol

# Debug del daemon
tailscale debug derp-map                  # imprime DERP map completo del control plane
tailscale debug component-logs magicsock --for=30m   # logs verbosos NAT traversal
tailscale debug component-logs derp --for=30m        # logs cliente DERP
tailscale debug daemon-logs               # stream logs daemon via LocalAPI
tailscale debug control-knobs             # NodeAttrs activos (incluido only-tcp-443)
tailscale debug restun                    # forzar nuevo ciclo STUN
tailscale debug rebind                    # forzar magicsock rebind sockets UDP
tailscale debug force-prefer-derp <region-id>  # forzar región DERP (ej: 17 = Miami)
tailscale bugreport                       # genera reporte completo para soporte

# Variables de entorno para forzar comportamiento
sudo TS_DEBUG_ALWAYS_USE_DERP=1 tailscaled --state=/var/lib/tailscale/tailscaled.state
sudo TS_DEBUG_NEVER_DIRECT_UDP=true tailscaled    # simula NAT simétrico para testing
sudo TS_DEBUG_NETCHECK=1 tailscaled               # netcheck verboso

# Logs systemd
sudo journalctl -u tailscaled -f --no-pager
sudo journalctl -u tailscaled --since="1 hour ago" | grep -i "derp\|udp\|relay\|blocked\|error"
```

**Repositorios alternativos analizados:**

| Tool | Stars | ARM64 | Self-host control | Funciona con UDP block | Costo recurrente |
|---|---|---|---|---|---|
| `tailscale/tailscale` | 32k | Sí | Vía Headscale | Sí (DERP TCP 443) | Gratis ≤100 devices |
| `juanfont/headscale` | 38.7k | Sí (`headscale_*_linux_arm64.deb`) | Sí | Sí (con DERP propio en VPS) | VPS ~5 USD/mes |
| `cloudflare/cloudflared` | 14.3k | Sí (binario estático) | No (Cloudflare edge) | Sí (HTTP/2 over TCP 443, QUIC opcional) | Gratis para SSH/TCP |
| `fatedier/frp` | 106.7k | Sí | Sí (necesita VPS) | Sí (TCP puro, configurable en :443) | VPS ~5 USD/mes |
| `rapiz1/rathole` | 13.6k | Sí (musl) | Sí | Sí | VPS ~5 USD/mes |
| `ekzhang/bore` | 11.2k | Sí (musl) | Opcional (bore.pub público) | Sí | Gratis con bore.pub |
| ngrok TCP | — | Sí | No | Sí | Free: URL aleatoria. Pro: 10 USD/mes |
| WireGuard puro | — | Sí (built-in kernel Ubuntu 18.04 backports) | Sí | No (es UDP, requiere UDP abierto al VPS o `udp2raw`/`wstunnel`) | VPS ~5 USD/mes |

### A.3 Video — universo POPULAR (canales >100k subs)

Cobertura del estrato top en inglés, 7 videos analizados. Hallazgo clave: **ningún video popular cubre el caso de Tailscale bloqueado en universidad/corporativo**. Lo que existe es contenido sobre features generales y exit nodes. La brecha del estrato popular es real.

Videos más relevantes:
- **"7 Essential Tailscale CLI Commands"** (Tailscale oficial, 14k views, 2025) — `https://youtu.be/k3NqliNGo6s` — capítulo 23:31 dedicado a `tailscale netcheck`.
- **"Tailscale Is Awesome — DERP vs Local"** (Jim's Garage, 36k views, 2024) — `https://youtu.be/53hqWTUkogk?t=724` — benchmark empírico DERP vs direct: degradación significativa de throughput cuando cae a DERP.
- **"Your Tailscale Has the Wrong Derpmap!"** (TECH EXTRANET, 7.7k views, 2025) — `https://youtu.be/XOD3yfQKHoo` — único video que muestra cómo restringir DERP map vía ACL JSON.

### A.4 Video — universo MID (10k-100k subs)

7 videos analizados, foco SRE/network engineers. Hallazgos clave:

- **"Get full line speed with Tailscale Peer Relays"** (Tailscale oficial, 46k views, 2025) — `https://youtu.be/wkBSjT1hO6k` — anuncia **Peer Relays** GA. Demo cuantitativa: 968 MB/s direct vs 38 MB/s DERP forzado (`TS_DEBUG_NEVER_DIRECT_UDP=true`). **Peer Relays NO usan TCP 443**, así que NO sirven para sobrevivir bloqueo en redes restrictivas — solo son alternativa para throughput cuando DERP es lento.
- **"VPN Without Port Forwarding Using Headscale"** (Jim's Garage, 47k views, 2023) — `https://youtu.be/u_6Zd7Bo6J4` — tutorial 37 min de Headscale en Oracle Free Tier VPS.
- **"NetBird vs Tailscale"** (45HomeLab, 52k views) y **"Comparing Overlay VPN Networks"** (Lawrence Systems, 173k views) — comparativas estructuradas; consenso: Tailscale + Headscale es la combinación más madura.

### A.5 Video — universo NICHE / LATAM long-tail

**Gap confirmado y documentado:** No existe ningún video en YouTube en español/portugués/inglés que documente específicamente el bloqueo de Tailscale en universidades colombianas. Cero resultados con queries combinando "Colombia", "UAO", "Universidad Autónoma" + Tailscale/WireGuard.

Videos relevantes en español:
- **"Tailscale o WireGuard para VPN en tu VPS? AMBOS!"** (Jonatan Castro, España, 14k views) — `https://youtu.be/cxHwVsgVKRA?t=645` — demuestra Tailscale como exit node detrás de VPS Hetzner (~5 EUR/mes), patrón aplicable.
- **"ROMPE EL CGNAT: VPN a Tu Casa con WireGuard"** (Dev Knives, España, 39k views) — `https://youtu.be/faKYN92QTUU` — patrón de VPS como relay intermediario, análogo a DERP.

**Comunidades LATAM activas sobre Tailscale: ausentes.** El ecosistema de selfhosting en español está mayoritariamente producido desde España, no LATAM.

---

## Track B — Búsqueda ampliada y lectura profunda

### B.1 Documentación oficial Tailscale (canónica)

**KB 1082 — Firewall ports** (`https://tailscale.com/kb/1082/firewall-ports`):

Requisitos exactos para que Tailscale funcione plenamente:

| Regla | Puerto | Destino | Función |
|---|---|---|---|
| TCP outbound | `*:443` | cualquiera | Control plane + DERP relays (HTTPS) |
| UDP outbound | `:41641 → *:*` | cualquiera | WireGuard directo P2P |
| UDP outbound | `*:3478` | cualquiera | STUN (detección NAT type) |
| TCP outbound | `*:80` | cualquiera | Control plane preferido (más eficiente) + captive portal detection |

**Rangos IP estáticos** (julio 2025+) para firewalls que exigen reglas por IP:
- `login.tailscale.com` y `controlplane.tailscale.com`: IPv4 `192.200.0.0/24`, IPv6 `2606:B740:49::/48`
- `log.tailscale.com` (desde noviembre 2025): IPv4 `199.165.136.0/24`, IPv6 `2606:B740:1::/48`

**FQDNs canónicos del control plane** (si UAO permite allowlist por dominio):
- `console.tailscale.com`
- `controlplane.tailscale.com`
- `log.tailscale.com`
- `login.tailscale.com`
- DERP relays: `derp1-all.tailscale.com` … `derp28-all.tailscale.com` (a agosto 2025; rango crece)

**Curl directo del DERP map oficial:**
```bash
curl https://login.tailscale.com/derpmap/default | jq
```
Devuelve JSON con todos los DERP regions, hostnames, IPv4/IPv6 e IDs de región.

### B.2 KB 1457 — Captive portals

Detección activa desde Tailscale v1.72+ (PR #12707):

1. Cliente observa estado de interfaces y conexión al coordination server. Si detecta problema → arranca detección.
2. HTTP GET (no TLS) a relay servers conocidos por aceptar puerto 80, endpoint `/generate_204`.
3. Tailscale usa IPv4 directo (no DNS, porque DNS suele estar interceptado bajo captive portal).
4. Header de challenge: `X-Tailscale-Challenge: ts_<hostname>`. Respuesta esperada: status `204` + header `X-Tailscale-Response: response ts_<hostname>`.
5. Si el response es `204` con challenge correcto → no hay captive portal. Si no → captive portal probable.

Salida CLI: `tailscale status` incluye health check `"This network requires you to log in using your web browser."`

Notificaciones nativas solo en macOS/iOS. En Linux (Nano) hay que monitorear vía `tailscale status` o `journalctl`.

### B.3 KB hard-NAT issues

Cuando hay NAT simétrico ("hard NAT") los direct connections se vuelven imposibles → todo cae a DERP. Si hay packet rate alto sobre conexiones DERP TCP, riesgo de head-of-line blocking → TCP meltdown. Recomendación oficial Tailscale: **evitar hard NAT en exit nodes / subnet routers**. Para el caso del Nano (no es exit node), no aplica directamente.

### B.4 Schema.ai — insights estructurados sobre Tailscale failures

Schema.ai mantiene una base estructurada de modos de fallo de Tailscale. Tres insights relevantes:

1. **"UDP connectivity blocked prevents direct WireGuard connections forcing DERP relay"**
   `https://schema.ai/technologies/tailscale/insights/udp-firewall-block-forces-derp-relay`
   Diagnóstico: `tailscale netcheck` muestra `UDP: false`.
   Acción: verificar firewall outbound UDP 41641; si UDP queda bloqueado, aceptar DERP y verificar capacidad.

2. **"MappingVariesByDestIP indicates hard NAT preventing direct connections"**
   `https://schema.ai/technologies/tailscale/insights/mapping-varies-by-dest-ip-indicates-hard-nat`
   Si `MappingVariesByDestIP: true` → hard NAT → hole punching imposible. Soluciones: cambiar NAT del router a "easy NAT", reservar IP estática del Nano, o aceptar DERP.

3. **"Blocked UDP packets force fallback to relayed connections"**
   `https://schema.ai/technologies/tailscale/insights/blocked-udp-packets-force-fallback-relayed`
   Mismo síntoma desde otro ángulo: firewall/proveedor bloquea UDP.

### B.5 DeepWiki — `tailscale/tailscale` con pregunta dirigida

Síntesis de la respuesta AI-grounded del repo:

**Secuencia exacta del cliente Tailscale cuando UDP 41641 outbound está bloqueado:**

1. **STUN probes UDP** con timeout `stunProbeTimeout = 3s` (definido en `net/netcheck/netcheck.go`).
2. **HTTPS probing** a DERP servers (TCP 443) para medir latencia y descubrir DERP más cercano.
3. **ICMP probing** en paralelo (best-effort).
4. **DERP relay fallback**: tráfico WireGuard cifrado encapsulado sobre TLS/WebSocket TCP 443. Si no hay info de latencia (porque todo falló), elige DERP deterministico arbitrario.

**Campos del reporte `netcheck` (struct `Report` en `net/netcheck/netcheck.go`):**
- `UDP bool` — true si una STUN round trip completó
- `IPv6 bool` — análogo
- `MappingVariesByDestIP` — true si NAT asigna puerto distinto por destino (symmetric)
- `CaptivePortal` — true si HTTP probe a relay servers fue tampered
- `Nearest DERP` — region elegida; "unknown (no response to latency probes)" si todo falló
- `WorkingUDP` se propaga al control plane vía `tailcfg.NetInfo` (`tailcfg.go` línea 1045)

**Workarounds canónicos y a qué falla aplican:**

| Workaround | Falla que ataca | Cómo se aplica |
|---|---|---|
| `TS_DEBUG_ALWAYS_USE_DERP=1` | UDP totalmente bloqueado | Variable env al daemon → fuerza DERP-only, ahorra ciclos de retry |
| `NodeAttrOnlyTCP443` | Solo HTTPS permitido | Configurado desde control plane/Admin Console o headscale → cliente desactiva UDP+STUN+portmapper+peer-relay completo |
| Self-hosted `derper` | DERP oficial filtrado por DNS/SNI | `derper` con dominio propio + cert TLS válido → DERP map custom apunta a tu server |
| `tailscale up --port=<N>` | Solo UDP 41641 bloqueado (NO bloqueo UDP general) | Cambia puerto local WireGuard. **NO ayuda si bloqueo es por DPI o UDP global** |
| `tailscale up --accept-dns=false` | DNS de Tailscale entra en conflicto con red corporativa | Desactiva MagicDNS, usar IPs Tailnet directamente |

**Logs `magicsock` para diagnóstico:**
`tailscale debug component-logs magicsock --for=30m` muestra cada transición direct↔DERP, intento hole-punch, errores TLS, selección de path, cambios de DERP home.

### B.6 Self-hosted DERP — guía Janhouse

`https://www.janhouse.lv/blog/network/self-hosting-tailscale-derp-headscale`

Setup canónico para DERP propio (asumiendo VPS con dominio):

```yaml
# docker-compose.yml — fragmento clave
services:
  derp:
    image: janhouse/tailscaled-derper
    volumes:
      - ${CERTS}/fullchain.pem:/app/cert.crt
      - ${CERTS}/privkey.pem:/app/cert.key
      - /dev/net/tun:/dev/net/tun
    cap_add:
      - net_admin
    ports:
      - "0.0.0.0:3478:3478/udp"
    environment:
      DERP_CERT_MODE: "manual"
      DERP_DOMAIN: "${DOMAIN_NAME}"
      DERP_VERIFY_CLIENTS: "true"     # solo permite peers del tailnet
    labels:
      traefik.tcp.routers.derp-tcp.rule: HostSNI(`${DOMAIN_NAME}`)
      traefik.tcp.routers.derp-tcp.tls.passthrough: true   # SNI pass-through, NO termination
```

Puntos críticos:
- DERP corre en puerto 443 pero **NO es HTTPS plano**, es protocolo custom dentro de TLS.
- **SNI pass-through** (NO terminar TLS en el proxy) permite compartir puerto 443 con otros servicios HTTPS del VPS.
- DERP server necesita su **propio cert TLS válido** (Let's Encrypt funciona).
- `DERP_VERIFY_CLIENTS: true` requiere que el container también tenga un `tailscaled` corriendo dentro para listar peers del tailnet.
- **Ventaja para caso UAO**: el firewall ve solo tráfico HTTPS a `derp.tu-dominio.com` (no `*.tailscale.com`) → no debería matchear las categorías "VPN" / "remote access" de Fortinet.

### B.7 Issue #11789 — caso idéntico documentado

`https://github.com/tailscale/tailscale/issues/11789` (autor dolceAlka, abril 2024):

> "Tailscale has recently been blocked by fortiguard under the remote acccess category, which breaks tailscale usage by blocking access to the controlplane."

Solución reportada (parcial): usar el celular como tether para fetch del controlplane → conectar a exit node → ping a devices con direct connection. **Pero peers detrás de DERP siguen rotos** porque el dominio del DERP server también está categorizado y bloqueado.

Esto confirma que **categorización VPN de Fortinet/FortiGuard es probablemente lo que UAO usa**, ya que es la solución comercial más común en universidades colombianas (FortiGate aparece en licitaciones públicas de UAO, UPB, UNAL, UniAndes).

### B.8 BabaBuilds — guía 2026 NAT traversal

`https://bababuilds.com/blog/bypass-cgnat-remote-access-guide-2026/`

Benchmark consolidado de alternativas (1 Gbps line):

| Solución | Throughput | UDP / Gaming | CGNAT bypass | Self-host control |
|---|---|---|---|---|
| Netmaker (kernel WG) | ~950 Mbps | Excelente | Sí (auto) | Sí |
| WireGuard puro (kernel) | ~980 Mbps | Excelente | No (manual) | Sí |
| ZeroTier | ~600 Mbps | Bueno | Sí | Solo controller |
| Tailscale (userspace WG) | ~400 Mbps | OK | Sí (DERP) | Con Headscale |
| FRP (reverse proxy) | VPS-limited | Excelente | Sí (reverse) | Sí |
| Cloudflare Tunnel | ~200 Mbps | Pobre | Sí (outbound) | No |

Para el proyecto embebidos-3 (Jetson Nano, throughput SSH << 10 Mbps), la diferencia kernel vs userspace **no es relevante**. Lo crítico es **funcionalidad bajo bloqueo**.

### B.9 Papers académicos — DPI y VPN fingerprinting

Búsqueda en arXiv + ACM + IEEE Xplore retornó múltiples papers relevantes:

- **"OpenVPN is Open to VPN Fingerprinting"** (arXiv 2403.03998) — detección de OpenVPN. Para UDP/obfuscated servers, las sondas son menos efectivas → trasladable parcialmente a WireGuard.
- **"Efficacy of Full-Packet Encryption in Mitigating Protocol Detection"** (arXiv 2412.17352) — Full-Packet Encryption (FPE) hace el tráfico VPN indistinguible de ruido aleatorio. Es lo que **AmneziaWG** y otros forks de WireGuard implementan; relevante para FR #13119.
- **"Iran's Stealth Internet Blackout"** (arXiv 2507.14183) — modelo de DPI estatal, técnicas usadas extrapolables a firewalls universitarios sofisticados.
- **"Fingerprinting Deep Packet Inspection Devices by Their Ambiguities"** (arXiv 2509.09081) — caracterización de vendors DPI por comportamientos ambiguos. Útil para identificar qué firewall usa UAO si hay acceso a captura de paquetes.

---

## Análisis consolidado

### Cómo funciona Tailscale a nivel de red — modelo unificado

```
                    ┌─────────────────────┐
                    │ controlplane.       │  ← TCP 80 preferido / TCP 443 fallback
       ┌────────────│ tailscale.com       │     (auth, key exchange, netmap, DERP map)
       │            │ login.tailscale.com │
       │            └─────────────────────┘
       │                       │
       │                       │ HTTPS Noise protocol
       │                       │
   ┌───▼────┐              ┌───▼────┐
   │ Laptop │  ◄═══════════│  Nano  │   ← UDP 41641 (WireGuard directo)
   │        │   intento 1  │        │     STUN UDP :3478 para NAT discovery
   └───┬────┘              └───┬────┘
       │                       │
       │   intento 2 fallback  │
       │                       │
       │      ┌────────────┐   │
       └──────┤ DERP relay │───┘   ← TCP 443 (TLS+WebSocket, encapsula WG cifrado)
              │ derpN-all. │         derp.tailscale.com:443 → 1 de 28 regions
              │ tailscale  │
              │ .com       │
              └────────────┘
```

**Orden de operaciones del cliente al hacer `tailscale up`:**
1. TCP 80/443 a `controlplane.tailscale.com` → auth, recibe netmap + DERP map.
2. UDP `*:3478` (STUN) a DERPs → descubre IP pública + tipo de NAT.
3. Intenta hole punching UDP 41641 con cada peer.
4. Si falla → conexión persistente TCP 443 al DERP home → tráfico encapsulado.
5. Reintentos periódicos de upgrade direct (cada ~30s) hasta lograr peer-to-peer.

### Vectores de bloqueo en redes corporativas — taxonomía

| Vector | Implementación típica | Detección desde el cliente | Vencible con... |
|---|---|---|---|
| **DNS filtering** de `*.tailscale.com` | Resolver corporativo + blocklist | `nslookup controlplane.tailscale.com` retorna NXDOMAIN o IP falsa | DNS-over-HTTPS (DoH) en el cliente, o DERP propio con dominio neutral |
| **IP blocking** de rangos Tailscale | Firewall L3 con ACL por IP | `curl -v https://192.200.0.x` falla con timeout | DERP propio en VPS con IP no categorizada |
| **Application Control / SNI inspection** | Fortinet App-ID, Palo Alto App-ID, SonicWall, Cisco Umbrella | Conexión TLS a `tailscale.com` se cierra mid-handshake; `tailscale netcheck` muestra "Nearest DERP: unknown" | DERP propio con SNI neutral (NO `*.tailscale.com`) |
| **UDP outbound deny-all** (excepto 53/123) | Firewall L4 default deny | `tailscale netcheck` muestra `UDP: false` | DERP TCP 443 funciona; aceptar latencia |
| **DPI fingerprinting de WireGuard** (148-byte handshake) | Fortinet/Palo Alto App-ID con signature WireGuard | Conexión WG directa falla aunque UDP 41641 esté abierto | DERP only (encapsula WG en TLS, oculta firma) |
| **Captive portal sin completar** | HTTPS intercept antes de auth | `tailscale netcheck` muestra `CaptivePortal: true` | Completar auth en navegador, luego activar Tailscale |
| **Symmetric NAT corporativo** | NAT gateway empresarial | `MappingVariesByDestIP: true` | Inevitable, forzar DERP |
| **TLS MITM / SSL inspection** | Proxy corporativo termina TLS y re-cifra | Errores TLS al `tailscale netcheck`; cert no es de Tailscale | Excepción del proxy para dominios Tailscale, o DERP propio con cert pinned por cliente |

### Aplicación al caso UAO — hipótesis priorizadas (orden de probabilidad)

**Hipótesis 1: DNS filtering (probabilidad ALTA)**
UAO usa resolver DNS institucional con blocklist de categorías comerciales (típicamente Cisco Umbrella, FortiGuard, Bluecat). Categoría "Anonymizer/VPN" incluye Tailscale.
- **Síntoma esperado**: `nslookup controlplane.tailscale.com` desde laptop UAO retorna NXDOMAIN o IP falsa.
- **Tailscale netcheck**: probablemente falla en arrancar o muestra DERP unknown.
- **Fix barato**: DoH en el cliente (`tailscale up --accept-dns=true` con DNS Tailscale, o cliente DoH local en el Nano apuntando a `1.1.1.1`/`9.9.9.9`).

**Hipótesis 2: FortiGate con App-ID categorizando Tailscale (probabilidad ALTA)**
UAO casi seguro usa FortiGate (estándar en universidades colombianas). Issue #11789 documenta exactamente este patrón.
- **Síntoma esperado**: DNS resuelve OK pero handshake TLS a `controlplane.tailscale.com` se cierra; alternativamente UDP WireGuard se bloquea aunque puerto 41641 esté abierto.
- **Fix**: DERP propio en VPS con dominio neutral (no `*.tailscale.com`) + SNI passthrough.

**Hipótesis 3: UDP outbound deny-all (probabilidad MEDIA-ALTA)**
Política conservadora estándar en universidades sin recursos para granularidad: deny UDP excepto 53 (DNS) y 123 (NTP).
- **Síntoma esperado**: `tailscale netcheck` muestra `UDP: false`. Pero **DERP TCP 443 debería funcionar** sin más config.
- **Fix**: ninguno necesario (Tailscale ya hace fallback automático). Aceptar la latencia de DERP Miami (~88ms medidos).

**Hipótesis 4: Symmetric NAT corporativo (probabilidad MEDIA)**
NAT gateway de UAO podría asignar puerto distinto por destino → hole punching imposible aunque UDP esté permitido.
- **Síntoma esperado**: `MappingVariesByDestIP: true` en netcheck.
- **Fix**: aceptar DERP. Si Peer Relay funciona desde el laptop con buena conectividad → throughput mejorado.

**Hipótesis 5: TP-Link en modo cliente con WPA2-Enterprise mal configurado (probabilidad BAJA-MEDIA)**
APs TP-Link consumer (TL-WR841N, Archer C50, etc.) en modo WISP/Client Bridge tienen problemas conocidos con WPA2-Enterprise (PEAP/EAP-TTLS). Si UAO es Enterprise, el AP podría fallar el handshake periódicamente.
- **Síntoma esperado**: conexión intermitente, no falla total. Otros servicios HTTPS también afectados.
- **Fix**: usar un router que sí soporte Enterprise (Ubiquiti, MikroTik) en modo cliente, o configurar Enterprise directamente en el Nano via NetworkManager.

---

## Checklist de diagnóstico in-situ en UAO

Cuando se llegue físicamente a la UAO con el setup (laptop + AP TP-Link + Nano), ejecutar en orden:

### Paso 1 — verificar resolución DNS (desde laptop conectada a UAO)

```powershell
# Windows PowerShell
Resolve-DnsName controlplane.tailscale.com
Resolve-DnsName login.tailscale.com
Resolve-DnsName derp1-all.tailscale.com
Resolve-DnsName derp17-all.tailscale.com  # Miami
```

**Interpretación**: si retorna NXDOMAIN, IP `0.0.0.0`, IP de UAO o IP claramente falsa → **Hipótesis 1 confirmada**.

### Paso 2 — verificar conectividad TCP a control plane

```powershell
Test-NetConnection controlplane.tailscale.com -Port 443
Test-NetConnection login.tailscale.com -Port 443
```

**Interpretación**:
- `TcpTestSucceeded: True` → control plane TCP 443 alcanzable.
- `TcpTestSucceeded: False` → IP blocked O firewall L4 bloquea. Cruzar con `Resolve-DnsName` previo.

### Paso 3 — verificar HTTPS handshake (detecta SSL inspection / SNI block)

```powershell
curl.exe -v --max-time 10 https://controlplane.tailscale.com/ 2>&1 | Select-String -Pattern "subject|issuer|HTTP/|SSL|TLS"
curl.exe -v --max-time 10 https://derp17.tailscale.com/derp/probe 2>&1 | Select-Object -First 30
```

**Interpretación**:
- Si `issuer` no es Let's Encrypt/Cloudflare/DigiCert sino algo como "UAO" o "Fortinet CA" → **TLS MITM activo**.
- Si conexión se corta con `TLS handshake failed` → **SNI block confirmado** (Hipótesis 2).
- Si `HTTP/2 200` con cert legítimo → control plane funcional.

### Paso 4 — Tailscale netcheck (laptop)

```powershell
tailscale netcheck
```

Output esperado en escenario sano:
```
* UDP: true
* IPv4: yes, <ip-pública-UAO>:<puerto>
* MappingVariesByDestIP: false
* CaptivePortal: false
* Nearest DERP: <region>: <latency>ms
```

**Tabla de diagnóstico por output:**

| `UDP` | `MappingVaries` | `CaptivePortal` | `Nearest DERP` | Diagnóstico |
|---|---|---|---|---|
| true | false | false | <region>:<latency> | Red OK, Tailscale debería funcionar P2P direct |
| true | **true** | false | <region>:<latency> | Symmetric NAT → DERP forzado, latencia degradada |
| **false** | n/a | false | <region>:<latency> | UDP bloqueado, DERP funciona (DERP-only mode) |
| **false** | n/a | false | **unknown** | UDP bloqueado **Y** DERP TCP 443 también filtrado → bloqueo total |
| n/a | n/a | **true** | n/a | Captive portal sin completar → autenticar en browser |

### Paso 5 — verificar estado del Nano (si tiene WiFi reachable de otra forma, ej: SSH via cable a otro laptop UAO)

```bash
# En el Nano via consola física o método alterno
tailscale netcheck --verbose
tailscale status
sudo journalctl -u tailscaled -n 200 --no-pager | grep -i 'derp\|udp\|error\|relay'
```

### Paso 6 — Tailscale ping cruzado (si Nano sí está online)

```powershell
tailscale ping --verbose 100.100.166.120
```

Output esperado:
- `pong from jetson-nano (100.100.166.120) via DERP(<region>) in <X>ms` → DERP-only path, funciona pero lento.
- `pong from jetson-nano (100.100.166.120) via <ip-publica>:<puerto> in <X>ms` → P2P direct (ideal).
- `timeout` → conexión rota, ver paso 4.

### Paso 7 — generar bugreport oficial (si quieres reportar a Tailscale o documentar)

```powershell
tailscale bugreport "Bloqueo en red WiFi UAO 2026-05-XX"
```
Genera un identificador opaco que se puede compartir con Tailscale support.

---

## Workarounds priorizados — orden de viabilidad para el caso

### Tier 1 — sin infra adicional ni costos (probar PRIMERO en UAO)

**W1. Aceptar DERP fallback** — si solo UDP está bloqueado pero DERP TCP 443 funciona.
- Acción: ninguna config extra. Tailscale ya hace fallback automático.
- Verificación: `tailscale ping` muestra "via DERP". SSH al Nano sigue funcionando (más lento).
- Latencia esperada: 80-150 ms (Miami DERP).
- **Para SSH es perfectamente usable.**

**W2. Forzar DERP-only modo** — si UDP intermitente causa retries que ralentizan más que aceptar DERP de entrada.
- Acción en el Nano: `sudo systemctl edit tailscaled` → añadir `Environment="TS_DEBUG_ALWAYS_USE_DERP=1"` → `sudo systemctl restart tailscaled`.
- Reverte: eliminar la línea cuando vuelvas a red residencial.

**W3. `--accept-dns=false`** — si MagicDNS entra en conflicto con resolver UAO.
- Acción: `sudo tailscale up --accept-dns=false`
- Útil si en UAO los nombres `*.tail091a20.ts.net` rompen pero IPs Tailnet directas (100.100.166.120) funcionan.

**W4. Hotspot 4G del celular** — plan B garantizado.
- Acción: compartir datos 4G del celular como hotspot WiFi. Conectar el TP-Link (o laptop directamente) al hotspot en vez de a UAO.
- Costo: datos móviles del plan.
- **Funciona con 100% de probabilidad** porque bypasa toda la red UAO.
- **Recomendado para sustentación**: tenerlo listo como backup independientemente del workaround principal.

### Tier 2 — requiere infra adicional, preparación previa antes de UAO

**W5. DERP propio en VPS con dominio neutral**
- Componentes: VPS público (Hetzner, DigitalOcean, Oracle Free Tier) + dominio + cert TLS Let's Encrypt.
- Setup ~30 min (sigue guía Janhouse).
- **Crítico**: configurar en Admin Console de Tailscale el DERP map custom apuntando al server propio.
- Funciona contra: SNI block de `*.tailscale.com`, IP block de Tailscale, App-ID corporativo categorizando "tailscale".
- NO funciona contra: bloqueo de control plane (si `controlplane.tailscale.com` está caído, ni Headscale resuelve).

**W6. Headscale self-hosted (control plane completo)**
- Reemplaza el control plane de Tailscale Inc. con tu propio servidor.
- Cliente Tailscale apunta con `tailscale up --login-server=https://headscale.tu-dominio.com`
- Funciona contra: bloqueo total de `*.tailscale.com` (incluido control plane).
- Setup ~1 hora (Headscale + Postgres + Caddy/Traefik + Let's Encrypt).
- Bonus: si Headscale tiene DERP embedded habilitado, no necesitas un derper aparte.
- **Recomendado para dependencia 0 de infra Tailscale**.

**W7. Cloudflare Tunnel (cloudflared) como alternativa**
- Reemplaza Tailscale completo para acceso SSH al Nano.
- Setup: instalar `cloudflared-linux-arm64` en el Nano + cuenta Cloudflare gratuita + dominio Cloudflare (gratis con cuenta).
- Comando único: `cloudflared tunnel --url ssh://localhost:22` + DNS record automático.
- Funciona contra: cualquier red que permita HTTPS outbound (es decir, cualquiera viable). Usa HTTP/2 sobre TCP 443, indistinguible de tráfico HTTPS normal.
- **NO** soporta UDP (no transferir datos UDP de sensores).
- Para SSH puro al Nano: 0 problemas.

### Tier 3 — workarounds de último recurso (no recomendados para MVP)

**W8. WireGuard sobre UDP 53 / UDP 123** (puertos DNS/NTP suelen estar abiertos)
- Configurar WireGuard puro (no Tailscale) escuchando en puerto 53 UDP en el VPS.
- Cliente Nano conecta a `vps.tu-dominio.com:53`.
- Riesgo: UAO probablemente intercepta UDP 53 con su propio resolver (NAT64 / DNS proxy).
- Reportado en Level1Techs sin éxito documentado.

**W9. WireGuard tunneled over TCP** (`udp2raw`, `wstunnel`)
- Encapsular WireGuard en TCP/443 para evadir DPI UDP.
- Complejidad alta, latencia degradada, no compatible con stack Tailscale stock.

**W10. SSH directo via cloudflared access** (`cloudflared access ssh`)
- Equivalente a W7 pero con Zero Trust de Cloudflare.
- Más config (registro de identidad), menos overhead.

---

## Plan de acción recomendado

### Antes de ir a la UAO (preparación)

1. **Pre-instalar `cloudflared-linux-arm64` en el Nano** como backup independiente:
   ```bash
   wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64
   chmod +x cloudflared-linux-arm64
   sudo mv cloudflared-linux-arm64 /usr/local/bin/cloudflared
   ```
   Configurar túnel de prueba desde casa para validar que funciona.

2. **Habilitar `tailscale up --ssh`** en el Nano (Tailscale SSH integrado, no requiere claves separadas) — útil si la auth se enreda.

3. **Tener bugreport limpio del estado normal**: `tailscale bugreport "baseline-casa-nicolas-2026-05-XX"`.

4. **Backup config NetworkManager del Nano** por si el TP-Link falla y hay que conectar Ethernet directo:
   ```bash
   sudo cp -r /etc/NetworkManager/system-connections /home/jetson/backups/nm-2026-05-XX/
   ```

5. **Probar localmente forzar DERP** con `TS_DEBUG_ALWAYS_USE_DERP=1` y medir latencia SSH — para confirmar que es usable antes de que sea emergencia.

6. **Decidir si vale la pena montar DERP propio AHORA** (preventivo, ~30 min + VPS ~5 USD/mes) o esperar a confirmar bloqueo en UAO (más eficiente económicamente, más riesgo en sustentación).

### Día de la sustentación en UAO

1. Llegar 30-45 min antes para diagnóstico.
2. **Ejecutar checklist completo** del paso 1 al 7 (~10 min total).
3. Según resultado:
   - **Todo OK** → proceder.
   - **Solo UDP bloqueado, DERP funciona** → proceder con latencia mayor (avisar audiencia que SSH va lento por DERP).
   - **DERP también bloqueado** → activar plan B (hotspot 4G).
   - **Captive portal** → autenticar en browser primero.
4. Mantener hotspot 4G como fallback durante toda la sustentación.

### Después (lessons learned para `mnemon`)

Guardar en memoria:
- Cuál hipótesis fue la correcta (confirmado empíricamente).
- Cuál workaround funcionó.
- Latencias y comportamientos observados.
- Lessons sobre firewall específico de UAO (modelo, vendor) si se identificó.

---

## Tabla resumen de alternativas — decision matrix para embebidos-3

| Solución | Setup | Costo | Funciona si UAO bloquea... | Recomendación para embebidos-3 |
|---|---|---|---|---|
| Tailscale stock (default) | 0 (ya está) | Gratis | Solo UDP | ✅ Probar primero. Acepta DERP. |
| Tailscale + `--accept-dns=false` | 1 min | Gratis | Conflicto DNS | ✅ Workaround barato si MagicDNS falla |
| Tailscale + DERP propio en VPS | 30-60 min | ~5 USD/mes | SNI/IP de `*.tailscale.com` | ⚠️ Considerar si bloqueo confirmado |
| Headscale completo | 1-2 hr | ~5 USD/mes | TODO `*.tailscale.com` | ⚠️ Sobreingeniería para MVP |
| Cloudflare Tunnel | 15 min | Gratis | Cualquier cosa (HTTPS outbound) | ✅ **Backup ideal para SSH** |
| frp / rathole | 30 min | ~5 USD/mes VPS | UDP+SNI | ⚠️ Si necesitas UDP custom |
| Hotspot 4G | 0 | Datos móviles | TODO red UAO | ✅ **Plan B garantizado para sustentación** |

---

## Casos LATAM / Colombia — gap confirmado

Búsqueda exhaustiva en YouTube + Reddit + foros técnicos + blogs no retornó **ningún caso documentado público** de Tailscale bloqueado específicamente en:
- Universidad Autónoma de Occidente (UAO)
- Otras universidades colombianas (UPB, UniAndes, UNAL, EAFIT, ICESI, Javeriana)
- Empresas o gobierno colombiano

**Interpretación**: la comunidad hispanohablante de homelab es pequeña; los reportes informales viven en Telegram/WhatsApp/Discord no indexados. La ausencia NO significa que el bloqueo no exista — significa que somos pioneros documentando este caso. **Vale considerar publicar los hallazgos** (Reddit r/Tailscale o blog técnico personal) como contribución a la comunidad.

---

## Notas sobre la investigación

- **Profundidad declarada**: ALTO. Resultado: 5 agentes Track A + 3 queries discover.py Track B + 8 lecturas Exa crawl + 1 DeepWiki = ~120 fuentes únicas tocadas, ~20 leídas en profundidad.
- **AAI fallback de transcripts**: NO activado (costo evitado).
- **Quota YouTube usada**: ~2.400 unidades (3 agentes video con queries diversas).
- **Tiempo aproximado**: 30 min agentes en paralelo + 10 min consolidación.

---

## Ronda 2 — 2026-05-23 (validación empírica in-situ)

### Diagnóstico real ejecutado en Wi-Fi UAO

Datos crudos capturados (laptop conectado directamente a SSID `WiFi-UAO`, BSSID `90:4c:81:98:9f:30`, IP local `11.11.13.49`, gateway `11.11.13.1`):

| Test | Resultado | Interpretación |
|---|---|---|
| DNS `*.tailscale.com` | ✅ Resuelve a IPs canónicas (`192.200.0.x`, `2606:b740:49::x`) | **H1 DNS filtering DESCARTADA** |
| TCP 443 `controlplane.tailscale.com` | ✅ `TcpTestSucceeded: True` | Control plane TCP alcanzable |
| TCP 443 `login.tailscale.com` | ✅ `TcpTestSucceeded: True` | Login endpoint TCP alcanzable |
| `tailscale netcheck` UDP | ✅ `true` | **H3 UDP block DESCARTADA** |
| `tailscale netcheck` `MappingVariesByDestIP` | ⚠️ **`true`** | **H4 Symmetric NAT CONFIRMADA** |
| `tailscale netcheck` CaptivePortal | ✅ `false` | Sin portal cautivo |
| `tailscale netcheck` Nearest DERP | Miami 61 ms | DERP alcanzable |
| IP pública vista (STUN) | `45.5.191.228:57789` | NAT outbound funciona |
| `tailscale status` Self | `offline` | **Control plane handshake degradado** |
| `tailscale status` Nano peer | `active; relay "mia"` | **Data plane funcionando vía DERP** |
| Health warning | `Network equipment from "Fortinet" may be blocking Tailscale traffic` | **H2 FortiGate CONFIRMADA por el propio cliente** |
| `tailscale ping nano` | `pong via DERP(mia) in 122-160ms` | Túnel operativo, P2P direct imposible |

### El descubrimiento clave — bypass por orden de conexión

El usuario reportó (y se reprodujo): conexión inicial a `WiFi-UAO` **falla** (Tailscale no autentica), pero al alternar primero a datos móviles 4G y autenticar Tailscale, y luego regresar a `WiFi-UAO`, **el túnel sigue vivo vía DERP**.

**Explicación técnica:** FortiGate App-ID detecta el handshake inicial de Tailscale por firma SNI/protocolo en `controlplane.tailscale.com` y lo bloquea (categoría "VPN" / "Remote Access"). Una vez establecida la sesión TCP/443 a un DERP relay (IPs como `199.38.181.x`, `208.111.40.x`), el firewall NO inspecciona el contenido (paquetes WireGuard cifrados encapsulados en TLS) y lo deja pasar como HTTPS genérico. El cliente Tailscale mantiene la sesión DERP indefinidamente mientras tenga conectividad TCP/443.

### Workaround empírico validado (orden de operaciones)

1. **Antes de entrar a UAO** (o al cambiar de red): conectar laptop a datos móviles del celular como hotspot WiFi.
2. **Ejecutar** `tailscale up` y completar cualquier auth check pendiente (URL `https://login.tailscale.com/a/<code>` abierta en browser).
3. **Verificar** con `tailscale status` que el Nano aparezca `active`.
4. **Conmutar** a `WiFi-UAO`. La sesión Tailscale sobrevive el switch de red porque solo necesita TCP/443 outbound, que FortiGate no inspecciona.
5. **Operar normalmente** vía Tailscale: SSH al Nano, dashboard `ws://100.100.166.120:8000/ws`, etc.

Si el cliente Tailscale cae offline en mitad de operación, repetir desde paso 1.

### Confirmación dashboard funcional en UAO

Estado del dashboard `embebidos-3 live detection` con el workaround aplicado:
- WebSocket: `ws://100.100.166.120:8000/ws` (IP Tailnet directa)
- Cámara: HD Pro Webcam C920
- Inferencia: detectando "plástico 63%" en pieza 3D-printed azul
- Métricas: 12 fps (target 14 fps), retardo 86 ms por frame, temperatura Nano 30 °C, memoria libre 1636 MB
- Estado: **conexión activa**

El retardo de 86 ms es latencia de procesamiento por frame (captura → JPEG → WS send → TRT inference → WS receive), no RTT puro. El WS reusa la misma conexión TCP/443, por lo que no paga handshake por frame.

### Hipótesis del documento original — veredicto empírico

| Hipótesis | Veredicto | Evidencia |
|---|---|---|
| H1: DNS filtering `*.tailscale.com` | ❌ Descartada | DNS resuelve a IPs canónicas |
| H2: FortiGate App-ID Tailscale | ✅ **Confirmada** | Cliente Tailscale reporta `Fortinet may be blocking` |
| H3: UDP outbound deny-all | ❌ Descartada | `UDP: true` en netcheck |
| H4: Symmetric NAT corporativo | ✅ **Confirmada** | `MappingVariesByDestIP: true` |
| H5: TP-Link 802.1X | N/A | Laptop directo a `WiFi-UAO`, sin TP-Link en el path |

**Conclusión definitiva:** UAO opera **bloqueo selectivo de Tailscale en la fase de autenticación** (App-ID Fortinet), no bloqueo total. Combinado con symmetric NAT corporativo, esto fuerza modo DERP-only pero NO impide el túnel si se autentica fuera de la red. El workaround empírico es trivial (auth en 4G, switch a UAO) y no requiere infra adicional (DERP propio, Cloudflare Tunnel, headscale).

---

## Historial de investigación

| Ronda | Fecha | Profundidad | Foco |
|-------|-------|-------------|------|
| 1 | 2026-05-22 | Alto | Diagnóstico y workarounds Tailscale bloqueado en UAO (teórico) |
| 2 | 2026-05-23 | Medio | Validación empírica in-situ en Wi-Fi UAO + descubrimiento del bypass por orden de conexión |

---

## Fuentes consultadas (acumulado)

| # | Título | URL | Tipo | Ronda |
|---|---|---|---|---|
| 1 | KB 1082 — Firewall ports | https://tailscale.com/kb/1082/firewall-ports | Doc oficial | 1 |
| 2 | KB 1457 — Captive portals | https://tailscale.com/kb/1457/captive-portals | Doc oficial | 1 |
| 3 | Troubleshoot hard NAT issues | https://tailscale.com/docs/reference/troubleshooting/network-configuration/hard-nat-issues | Doc oficial | 1 |
| 4 | Connection types | https://tailscale.com/docs/reference/connection-types | Doc oficial | 1 |
| 5 | NAT traversal pt.1 (blog) | https://tailscale.com/blog/nat-traversal-improvements-pt-1 | Blog oficial | 1 |
| 6 | NAT traversal pt.2 cloud | https://tailscale.com/blog/nat-traversal-improvements-pt-2-cloud-environments | Blog oficial | 1 |
| 7 | Peer Relays beta | https://tailscale.com/blog/peer-relays-beta | Blog oficial | 1 |
| 8 | DERP servers reference | https://tailscale.com/docs/reference/derp-servers | Doc oficial | 1 |
| 9 | Device connectivity | https://tailscale.com/docs/reference/device-connectivity | Doc oficial | 1 |
| 10 | Palo Alto firewall integration | https://tailscale.com/docs/integrations/firewalls/palo-alto-firewall | Doc oficial | 1 |
| 11 | Firewall integration | https://tailscale.com/docs/integrations/firewalls | Doc oficial | 1 |
| 12 | Issue #11789 — fortiguard remote access | https://github.com/tailscale/tailscale/issues/11789 | GitHub issue | 1 |
| 13 | Issue #11791 — DERP maps not stored locally | https://github.com/tailscale/tailscale/issues/11791 | GitHub issue | 1 |
| 14 | Issue #12456 — FR DERP latency unacceptable | https://github.com/tailscale/tailscale/issues/12456 | GitHub issue | 1 |
| 15 | Issue #13119 — FR WireGuard obfuscation | https://github.com/tailscale/tailscale/issues/13119 | GitHub issue | 1 |
| 16 | Issue #13810 — Eduroam | https://github.com/tailscale/tailscale/issues/13810 | GitHub issue | 1 |
| 17 | Issue #14287 — Diagnostics report no UDP | https://github.com/tailscale/tailscale/issues/14287 | GitHub issue | 1 |
| 18 | Issue #15217 — DERP and Fortinet | https://github.com/tailscale/tailscale/issues/15217 | GitHub issue | 1 |
| 19 | Issue #19259 — UDP false regression 1.96.4 | https://github.com/tailscale/tailscale/issues/19259 | GitHub issue | 1 |
| 20 | Issue #19748 — dialNodeUsingProxy hardcodes :443 | https://github.com/tailscale/tailscale/issues/19748 | GitHub issue | 1 |
| 21 | Issue #4377 — DERP latency MITM proxy | https://github.com/tailscale/tailscale/issues/4377 | GitHub issue | 1 |
| 22 | Issue #9524 — Tailscale can't work around blocked DERP | https://github.com/tailscale/tailscale/issues/9524 | GitHub issue | 1 |
| 23 | PR #12161 — Palo Alto DIPP workaround | https://github.com/tailscale/tailscale/pull/12161 | GitHub PR | 1 |
| 24 | PR #12707 — captive portal detection | https://github.com/tailscale/tailscale/pull/12707 | GitHub PR | 1 |
| 25 | PR #16517 — NodeAttrOnlyTCP443 | https://github.com/tailscale/tailscale/pull/16517 | GitHub PR | 1 |
| 26 | cmd/derper README | https://github.com/tailscale/tailscale/blob/main/cmd/derper/README.md | GitHub source | 1 |
| 27 | net/netcheck/netcheck.go | https://github.com/tailscale/tailscale/blob/main/net/netcheck/netcheck.go | GitHub source | 1 |
| 28 | cmd/tailscale/cli/netcheck.go | https://github.com/tailscale/tailscale/blob/main/cmd/tailscale/cli/netcheck.go | GitHub source | 1 |
| 29 | r/Tailscale — School Blocking (Fortinet) | https://www.reddit.com/r/Tailscale/comments/1kbqx5c/ | Foro | 1 |
| 30 | r/Tailscale — Fortinet blocking TS/WG | https://www.reddit.com/r/Tailscale/comments/1pm275j/ | Foro | 1 |
| 31 | r/Tailscale — SNI blocking DERP | https://www.reddit.com/r/Tailscale/comments/1g475fw/ | Foro | 1 |
| 32 | r/Tailscale — Self-hosted DERP circumvent | https://www.reddit.com/r/Tailscale/comments/1ha5wgs/ | Foro | 1 |
| 33 | r/Tailscale — No more DERP relays | https://www.reddit.com/r/Tailscale/comments/1iwhdnu/ | Foro | 1 |
| 34 | r/Tailscale — WG blocked work network | https://www.reddit.com/r/Tailscale/comments/1av3guf/ | Foro | 1 |
| 35 | r/Tailscale — Fortinet school wifi | https://www.reddit.com/r/Tailscale/comments/1kkphr8/ | Foro | 1 |
| 36 | Level1Techs — bypass firewall | https://forum.level1techs.com/t/so-how-would-you-bypass-a-firewall/229327 | Foro | 1 |
| 37 | HN — GoodbyeDPI WireGuard | https://news.ycombinator.com/item?id=32199468 | HN | 1 |
| 38 | Schema.ai — UDP block forces DERP | https://schema.ai/technologies/tailscale/insights/udp-firewall-block-forces-derp-relay | Insight | 1 |
| 39 | Schema.ai — MappingVariesByDestIP hard NAT | https://schema.ai/technologies/tailscale/insights/mapping-varies-by-dest-ip-indicates-hard-nat | Insight | 1 |
| 40 | Schema.ai — Blocked UDP forces relayed | https://schema.ai/technologies/tailscale/insights/blocked-udp-packets-force-fallback-relayed | Insight | 1 |
| 41 | Janhouse — Self-hosting DERP nodes | https://www.janhouse.lv/blog/network/self-hosting-tailscale-derp-headscale | Blog técnico | 1 |
| 42 | fexyn — DPI Explained | https://fexyn.com/blog/deep-packet-inspection-explained | Blog técnico | 1 |
| 43 | securetoolsguide — DPI detects VPN | https://securetoolsguide.com/vpn-packet-inspection-how-deep-packet-inspection-detects-vpn-traffic/ | Blog técnico | 1 |
| 44 | initez.nl — Understanding Tailscale | https://initez.nl/understanding-tailscale-a-modern-vpn-solution-for-secure-networking/ | Blog técnico | 1 |
| 45 | BabaBuilds — Ultimate Remote Access 2026 | https://bababuilds.com/blog/bypass-cgnat-remote-access-guide-2026/ | Blog técnico | 1 |
| 46 | tailscale-ips (jmaddington) | https://github.com/jmaddington/tailscale-ips | GitHub repo | 1 |
| 47 | awesome-tunneling | https://github.com/anderspitman/awesome-tunneling | GitHub repo | 1 |
| 48 | juanfont/headscale | https://github.com/juanfont/headscale | GitHub repo | 1 |
| 49 | rapiz1/rathole | https://github.com/rapiz1/rathole | GitHub repo | 1 |
| 50 | ekzhang/bore | https://github.com/ekzhang/bore | GitHub repo | 1 |
| 51 | fatedier/frp | https://github.com/fatedier/frp | GitHub repo | 1 |
| 52 | YouTube — 7 Essential CLI Commands | https://youtu.be/k3NqliNGo6s | Video | 1 |
| 53 | YouTube — Peer Relays | https://youtu.be/wkBSjT1hO6k | Video | 1 |
| 54 | YouTube — Tailscale Is Awesome | https://youtu.be/53hqWTUkogk | Video | 1 |
| 55 | YouTube — Lawrence Systems Comparing | https://youtu.be/eCXl09h7lqo | Video | 1 |
| 56 | YouTube — Level1Techs Avery Pennarun | https://youtu.be/UyczOQTx5Gg | Video | 1 |
| 57 | YouTube — Wrong Derpmap | https://youtu.be/XOD3yfQKHoo | Video | 1 |
| 58 | YouTube — Headscale Jim's Garage | https://youtu.be/u_6Zd7Bo6J4 | Video | 1 |
| 59 | YouTube — Jonatan Castro VPS+TS | https://youtu.be/cxHwVsgVKRA | Video | 1 |
| 60 | YouTube — Dev Knives CGNAT WG | https://youtu.be/faKYN92QTUU | Video | 1 |
| 61 | YouTube — Diolinux homelab | https://youtu.be/eCZW--yF1qg | Video | 1 |
| 62 | arXiv 2403.03998 — OpenVPN fingerprinting | https://arxiv.org/html/2403.03998v1 | Paper | 1 |
| 63 | arXiv 2412.17352 — Full-packet encryption | https://arxiv.org/html/2412.17352v1 | Paper | 1 |
| 64 | arXiv 2509.09081 — Fingerprinting DPI | https://arxiv.org/html/2509.09081v1 | Paper | 1 |
| 65 | arXiv 2507.14183 — Iran Stealth Blackout | https://arxiv.org/html/2507.14183v1 | Paper | 1 |
