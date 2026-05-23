# Deploy Headscale — control plane propio para bypass UAO

**Fecha:** 2026-05-23
**Objetivo:** Eliminar la fricción manual del bypass 4G→UAO. Reemplazar el control plane de Tailscale Inc. (`controlplane.tailscale.com`, bloqueado por FortiGate de UAO) por un Headscale self-hosted bajo un dominio que FortiGate no cataloga como VPN.

## Por qué funciona en UAO

| Componente | Tailscale Inc. (bloqueado) | Headscale (este deploy) |
|---|---|---|
| Control plane | `controlplane.tailscale.com` → SNI cae en categoría VPN → **bloqueado** | `80-241-217-130.nip.io` → no es `*.tailscale.com` → **pasa** |
| DERP (relay) | `derp*.tailscale.com` (americanos fallan incluso fuera de UAO) | **DERP embebido en el VPS** (region 999), puerto 443 + STUN 3478 |
| Data plane | WireGuard UDP / DERP | igual (WireGuard + DERP embebido) |

El control plane y el DERP viven en el **mismo hostname/puerto 443** que ya validamos que atraviesa el FortiGate.

## Infraestructura

**VPS Contabo de Frevalle** (`80.241.217.130`, Ubuntu 24.04, 4 vCPU, 7,8 GB RAM):
- Sitio Frevalle (nginx :80 + php8.2-fpm + MySQL) — **NO se tocó**. Puerto 443 estaba libre.
- Headscale + Caddy en Docker, aislados en `/opt/headscale/`.

```
/opt/headscale/
├── docker-compose.yml      # headscale (v0.28.0) + caddy (2-alpine)
├── config/config.yaml      # Headscale: server_url nip.io, DERP embebido on
├── Caddyfile               # TLS automático (Let's Encrypt TLS-ALPN-01) → reverse_proxy headscale:8080
├── data/                   # sqlite, noise key, DERP key
└── PREAUTH_KEY.txt         # pre-auth key reusable (chmod 600)
```

**Puertos abiertos en UFW del VPS** (aditivo, Frevalle intacto):
- 22, 80, 443 (preexistentes)
- **443/tcp+udp** → Caddy (control plane + DERP relay HTTPS)
- **41641/udp** → WireGuard directo
- **3478/udp** → STUN del DERP embebido

## Nodos (tailnet `embebidos3`)

| Nodo | IP Headscale | Rol |
|---|---|---|
| vps-frevalle | 100.64.0.1 | VPS (también corre el control plane) |
| nano-jetson | 100.64.0.2 | Jetson Nano del proyecto |
| (laptop) | 100.64.0.3 (al unirse) | Pendiente — lo conecta el usuario |

## Credenciales

- **Control plane:** `https://80-241-217-130.nip.io` (`/health` → `{"status":"pass"}`)
- **Pre-auth key reusable** (válida 1 año, en `/opt/headscale/PREAUTH_KEY.txt`):
  `hskey-auth-d4BQkuS34qpD-08wvIo3Jy5R2Dqb0UHZxL_WhqPEKIGJa_r2BbW2Eya3-0N9FMxVfEHIHvf6mGj84`
- **SSH al Nano:** ahora por llave (se agregaron `id_ed25519` del laptop y `frevalle_deploy`/`id_nano` del VPS al `authorized_keys`). El acceso ya no depende de Tailscale SSH.

## Migración del Nano (con dead-man-switch)

El Nano corría Tailscale 1.96.4 con `RunSSH: true`. Para migrar sin perder acceso:
1. Backup de `/var/lib/tailscale/tailscaled.state` (Tailscale Inc.).
2. Dead-man-switch (`systemd-run --on-active=12min`): restaura Tailscale Inc. si no se confirma éxito.
3. `tailscale up --login-server=https://80-241-217-130.nip.io --authkey=... --reset --force-reauth --accept-dns=false`.
   - **Clave:** cambiar de control plane exige `--force-reauth` (no solo `--reset`).

## Cómo conectar el laptop (desde UAO o cualquier red)

```powershell
tailscale up --login-server=https://80-241-217-130.nip.io --authkey=hskey-auth-d4BQkuS34qpD-08wvIo3Jy5R2Dqb0UHZxL_WhqPEKIGJa_r2BbW2Eya3-0N9FMxVfEHIHvf6mGj84 --accept-dns=false
```

Luego SSH al Nano por su IP Headscale:
```
ssh -i ~/.ssh/id_ed25519 jetson@100.64.0.2
```

## Estado de validación

- [x] Fase 1: Headscale + Caddy desplegados, cert TLS OK, control plane responde
- [x] Fase 2: usuario `embebidos3` + pre-auth key
- [x] Fase 3: VPS unido como nodo (100.64.0.1)
- [x] Fase 4: Nano migrado a Headscale (100.64.0.2) con --force-reauth
- [x] DERP embebido habilitado (region 999, IPs reales del VPS)
- [x] Fase 5: laptop unido (100.64.0.3) + SSH al Nano OK
- [ ] Validación desde UAO (la hace el usuario físicamente en el campus)

### Verificación en vivo (2026-05-23, post-migración)

VPS (`vps-headscale`):
- Headscale + Caddy `Up`; control plane `/health` → `{"status":"pass"}`.
- 3 nodos registrados y **ninguno expirado**: `vps-frevalle`, `jetson-desktop` (Nano), `desktop-3l89680` (laptop).
- DERP embebido escuchando: STUN `3478/udp` + relay `443`.
- **Frevalle intacto**: nginx activo, sitio HTTP 200, puerto 80 sigue siendo nginx (no docker).

Nano (`ssh nano` → `100.64.0.2`, ya sin `-F NUL`):
- `ControlURL: https://80-241-217-130.nip.io` (control plane propio).
- **Sin dead-man-switch pendiente** (no quedan timers `run-*`): el Nano es estable, no hay auto-rollback.
- Mesh visible: laptop con **conexión directa** (`direct 45.5.189.228:41641`, tráfico fluyendo).

Laptop:
- En Headscale como `100.64.0.3`; `ssh nano "hostname"` → `jetson-desktop` (HTTP 200 en FastAPI ya validado antes).
- `~/.ssh/config` con alias `nano` → `100.64.0.2`; permisos NTFS corregidos (sin `OWNER RIGHTS`).

> **Nota UAO:** en casa el camino laptop↔Nano es *directo* (UDP 41641 abierto). En UAO el FortiGate
> bloqueará 41641, forzando el **DERP embebido sobre `nip.io:443`** — que es precisamente el
> componente diseñado para atravesar el firewall. Esa es la prueba que falta hacer en el campus.

## Rollback total

```
cd /opt/headscale && docker compose down -v && rm -rf /opt/headscale
ufw delete allow 41641/udp && ufw delete allow 3478/udp
```
El VPS y Frevalle quedan exactamente como antes.
