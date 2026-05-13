# Acceso remoto y conectividad Wi-Fi — Jetson Nano B01 + Contabo VPC

**Proyecto:** `embebidos-3` (clasificador de residuos para Jetson Nano B01, entrega 2026-05-26).
**Dominio:** acceso remoto desde Windows 11 a la Nano (SSH + escritorio), conectividad Wi-Fi vía antena USB TP-Link TL-WN722N v4, AP mode standalone, y túneles overlay sobre VPC Contabo para alcanzar la Nano desde cualquier red.
**Documentos hermanos:** [`HANDOFF-implementacion-vastai-hf.md`](HANDOFF-implementacion-vastai-hf.md) · [`compatibilidad-stack-cloud-jetson.md`](compatibilidad-stack-cloud-jetson.md) · [`decisiones-D1-D15-ledger.md`](decisiones-D1-D15-ledger.md).
**Fecha de cierre Ronda 6:** 2026-05-12.

---

## 1. Resumen ejecutivo

El proyecto necesita acceso remoto persistente a la Nano para development (estas dos semanas) y para la entrega final (battery-powered, sin garantía de WiFi conocida). Tras esta ronda, el **stack recomendado** es:

| Componente | Elección | Razón principal |
|---|---|---|
| **SSH desde Win11** | OpenSSH built-in + ed25519 + `~/.ssh/config` | Sin instalación adicional, autenticación por clave, alias para puertos custom |
| **Escritorio remoto** | **NoMachine** + Xfce4 + **HDMI dummy plug** | Único que combina OpenGL accel (preview inferencia OpenCV), bajo overhead red (protocolo NX), cliente Win11 oficial, autostart headless |
| **Túnel overlay para acceso desde cualquier red** | **Tailscale free tier** (clientes oficiales en Nano + Win11) | Evita el bug WireGuard del kernel 4.9-tegra al usar `wireguard-go` (userspace), NAT traversal automático, setup en < 10 min |
| **Fallback si Tailscale Inc. falla** | Headscale self-hosted en Contabo | Mismo binario cliente, sólo cambia `--login-server` |
| **Driver Wi-Fi TL-WN722N v4** | `aircrack-ng/rtl8188eus@v5.3.9` vía DKMS | Único repo activo que compila en kernel 4.9 aarch64 con soporte AP mode |
| **Demo final sin WiFi conocido** | Nano en modo AP (hostapd + dnsmasq) **o** hotspot móvil + Tailscale | Dos rutas viables; preferir hotspot móvil por simplicidad |
| **Bastion en Contabo** | Reverse SSH autossh **+** Headscale opcional | Doble vía: tunnel persistente clásico + control plane Tailscale |

**Tres decisiones nuevas vinculantes** (D16-D20, ver §15) cierran esta ronda.

**Constraint descubierta crítica:** el módulo de kernel WireGuard NO funciona en JetPack 4.6.1 (bug en kernel 4.9-tegra, [foro NVIDIA #184764](https://forums.developer.nvidia.com/t/kernel-error-when-using-wireguard/184764)). Solución: usar implementación userspace (`wireguard-go` o Tailscale). **NO instalar `wireguard-dkms` directo en la Nano.**

---

## 2. Contexto

### 2.1 Hardware involucrado

| Componente | Especificación |
|---|---|
| Jetson Nano | Developer Kit 4 GB B01, JetPack 4.6.1, kernel 4.9.337-tegra, Ubuntu 18.04 bionic, aarch64 |
| Antena Wi-Fi USB | **TP-Link TL-WN722N v4** — chipset Realtek **RTL8188EUS** (USB ID típico `0bda:8179`) |
| Equipo de desarrollo | Windows 11 Pro (mitgar14) |
| Servidor bastion | Contabo VPC (asumido Ubuntu 20.04+/22.04, IP pública, acceso root SSH) |

**Crítico:** el TL-WN722N v1 usaba Atheros AR9271 (soporte mainline, monitor + AP). El **v4 usa RTL8188EUS** (Realtek), que requiere driver out-of-tree para AP mode estable. NO asumir specs del v1.

### 2.2 Escenarios de uso a cubrir

| Escenario | Cuándo aplica | Conectividad necesaria |
|---|---|---|
| **Development en casa/UAO** | Próximas 2 semanas | SSH directo a IP local + JupyterLab port-forward + opcional VNC para ajustes UI |
| **Acceso desde fuera (cualquier red)** | Cualquier momento, sobre todo si trabajo desde otro sitio | Túnel overlay (Tailscale) + opcional reverse SSH como respaldo |
| **Demo final battery-powered en UAO** | 2026-05-26 | Una de tres opciones: WiFi UAO + Tailscale, hotspot móvil + Tailscale, **o** Nano en modo AP con laptop conectado directo |

---

## 3. Antena TP-Link TL-WN722N v4 — chipset y limitaciones

### 3.1 Identificación

- **TL-WN722N v1**: Atheros AR9271 → soporte mainline kernel (`ath9k_htc`), monitor + AP nativos.
- **TL-WN722N v2 / v3 / v4**: Realtek **RTL8188EUS** → driver out-of-tree obligatorio para AP mode estable.

Verificar en la Nano tras conectar la antena:

```bash
lsusb | grep -i realtek
# Output esperado: Bus 001 Device 0XX: ID 0bda:8179 Realtek Semiconductor Corp. RTL8188EUS
```

### 3.2 Soporte AP confirmado

El RTL8188EUS **soporta modo AP** con el driver out-of-tree `aircrack-ng/rtl8188eus`, pero **NO** con el driver `r8188eu` mainline del kernel 4.9 (sólo soporta modo cliente). Restricciones del AP confirmadas en el [issue #261 del repo aircrack-ng/rtl8188eus](https://github.com/aircrack-ng/rtl8188eus/issues/261):

- Sólo banda 2,4 GHz (`hw_mode=g`, sin `ieee80211n`).
- Sólo WPA2-PSK (no WPA3, no SAE).
- NO usar `ieee80211w` (PMF).

### 3.3 La nota "DEPRECATED" del README NO aplica al RTL8188EUS

El README de `aircrack-ng/rtl8188eus` redirige a `lwfinger/rtw88`, pero ese repo es para chipsets **PCI/SDIO** (RTL8822BE/RTL8821CE), no USB. Para RTL8188EUS USB, el driver canónico vigente sigue siendo `aircrack-ng/rtl8188eus`, branch `v5.3.9`.

---

## 4. Driver RTL8188EUS para JetPack 4.6.1

### 4.1 Repo recomendado

**`aircrack-ng/rtl8188eus`** branch `v5.3.9` ([github.com/aircrack-ng/rtl8188eus](https://github.com/aircrack-ng/rtl8188eus)).

Razones:
- Único repo activo con DKMS soportado (`dkms-install.sh` incluido).
- `Makefile` mapea `aarch64 → arm64` automáticamente (no requiere edición para Nano).
- Issue #50 confirma compilación exitosa con `KVER=4.9.x` aarch64.
- Issue #261 documenta config hostapd que funciona para AP mode.

### 4.2 Dependencias y preparación del kernel source — CORRECCIÓN Ronda 7

> **Corrección R7:** la línea `sudo apt install linux-headers-$(uname -r)` **NO funciona en JetPack 4.6.x**. NVIDIA NO publica `linux-headers-X-tegra` en su repo APT. Confirmado por moderador NVIDIA `DaveYYY` en [foro #265546](https://forums.developer.nvidia.com/t/install-linux-headers-on-jetson-nano-failed/265546): *"kernel header files should already be included in `/usr/src/`"*. Ver §21 (Ronda 7) para receta funcional.

```bash
# Dependencias para compilar el driver (no headers via apt):
sudo apt-get update
sudo apt-get install -y git build-essential dkms bc \
    libssl-dev libelf-dev bison flex
```

**Headers/source del kernel:** se obtienen vía `getKernelSources.sh` + `make modules_prepare`. Receta completa en §21.

### 4.3 Instalación vía DKMS (persistente ante kernel updates)

```bash
cd /tmp
git clone https://github.com/aircrack-ng/rtl8188eus.git -b v5.3.9
cd rtl8188eus
sudo bash dkms-install.sh
```

El script ejecuta internamente:

```bash
sudo cp -r /tmp/rtl8188eus /usr/src/8188eu-5.3.9
sudo dkms add    -m 8188eu -v 5.3.9
sudo dkms build  -m 8188eu -v 5.3.9
sudo dkms install -m 8188eu -v 5.3.9
```

### 4.4 Blacklist del módulo mainline `r8188eu` (CRÍTICO)

Sin blacklist, el kernel puede cargar `r8188eu` (staging, sin AP) en lugar del `8188eu` out-of-tree (con AP):

```bash
sudo tee /etc/modprobe.d/realtek.conf <<'EOF'
blacklist r8188eu
blacklist rtl8xxxu
EOF
sudo update-initramfs -u
sudo reboot
```

### 4.5 Verificación post-instalación

```bash
lsmod | grep 8188eu          # esperado: 8188eu (SIN la r delante)
iw list | grep -A8 "Supported interface modes"  # debe listar: * AP
iwconfig wlan1               # o wlan0 según orden de detección
```

### 4.6 Power management — evitar desconexiones

```bash
sudo tee /etc/modprobe.d/8188eu.conf <<'EOF'
options 8188eu rtw_power_mgnt=0 rtw_enusbss=0 rtw_hwpwrp_detect=0 rtw_low_power=0 rtw_ips_mode=1
EOF

# NetworkManager:
sudo tee -a /etc/NetworkManager/NetworkManager.conf <<'EOF'

[connection]
wifi.powersave=2
EOF
sudo systemctl restart NetworkManager
```

---

## 5. AP mode con RTL8188EUS — configuración mínima validada

### 5.1 Restricciones inmutables del chipset

| Restricción | Justificación |
|---|---|
| Sólo 2,4 GHz banda `g` | El driver wext del RTL8188EUS no expone modo AP HT (802.11n) estable |
| Sólo WPA2-PSK + CCMP | WPA3/SAE producen `nl80211: Failed to set interface to mode 3: -95 (Operation not supported)` |
| Sin `ieee80211w` (PMF) | Mismo error que WPA3 |
| Throughput máx ~20-25 Mbps reales | Limitación de USB 2.0 + sin HT |

### 5.2 Config mínima de hostapd (verificada en issue #261)

`/etc/hostapd/hostapd.conf`:

```ini
interface=wlan1
driver=nl80211
ssid=embebidos3-nano
hw_mode=g
channel=6
wmm_enabled=0
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP
wpa_passphrase=<TuPasswordWPA2>
```

Apuntar systemd a este archivo:

```bash
sudo sed -i 's|#DAEMON_CONF=""|DAEMON_CONF="/etc/hostapd/hostapd.conf"|' /etc/default/hostapd
sudo systemctl unmask hostapd
sudo systemctl enable hostapd
```

---

## 6. NetworkManager hotspot — alternativa más simple (recomendada para demo)

Si `nmcli` reporta soporte AP en `iw list`, el setup es de una sola línea de comandos:

```bash
# Crear perfil hotspot:
sudo nmcli con add type wifi ifname wlan1 con-name embebidos3-hotspot \
  autoconnect yes ssid "embebidos3-nano"

# Configurar AP, WPA2, canal 6, subred IP:
sudo nmcli con modify embebidos3-hotspot \
  802-11-wireless.mode ap \
  802-11-wireless.band bg \
  802-11-wireless.channel 6 \
  802-11-wireless-security.key-mgmt wpa-psk \
  802-11-wireless-security.psk "<TuPasswordWPA2>" \
  ipv4.method shared \
  ipv4.addresses 192.168.42.1/24

# Activar:
sudo nmcli con up embebidos3-hotspot ifname wlan1
```

`ipv4.method shared` habilita automáticamente el DHCP server + masquerading. Perfil persistente en `/etc/NetworkManager/system-connections/embebidos3-hotspot`.

### 6.1 dnsmasq manual (si se evita NetworkManager)

`/etc/dnsmasq.conf`:

```ini
interface=wlan1
bind-interfaces
dhcp-range=192.168.42.10,192.168.42.100,255.255.255.0,24h
dhcp-option=3,192.168.42.1
dhcp-option=6,8.8.8.8,8.8.4.4
```

```bash
sudo systemctl enable dnsmasq
sudo systemctl start dnsmasq
```

---

## 7. Escritorio remoto — comparativa y recomendación

### 7.1 Tabla comparativa (Jetson Nano + JetPack 4.6.1)

| Criterio | vino / RealVNC bundled | x11vnc | TigerVNC | **NoMachine NX** | xrdp |
|---|---|---|---|---|---|
| Latencia WiFi 100 Mbps | Media | Media | Media-baja | **Baja** (protocolo NX) | Media |
| OpenGL hardware accel | Sí (display real) | Sí (display real) | No (sesión virtual) | **Sí** con dummy HDMI | **No** (EGL roto) |
| Autostart headless sin dummy | **No** (bug conocido) | Sí con workaround | Sí | Sí (mejor con dummy) | Sí |
| Cliente Win11 | VNC Viewer / TightVNC | VNC Viewer | TigerVNC viewer | **App oficial NX** | Remote Desktop nativo |
| RAM idle en Nano | ~30 MB | ~15 MB | ~20 MB | ~80-100 MB (con Xfce: ~40-50) | ~25 MB |
| Bugs JetPack 4.6.1 | Sin HDMI no arranca | Resolución incorrecta sin dummy | Sesión nueva sin CUDA | Alto CPU si GNOME (Xfce lo arregla) | **Crash inmediato + EGL roto** |

### 7.2 Bugs específicos confirmados

- **xrdp descartado**: moderador NVIDIA `linuxdev` confirma que "EGL does not work with rdp backend" → rompe OpenGL/CUDA en la sesión RDP ([forums.developer.nvidia.com/t/xrdp-login-profile-different-from-boot-with-monitor/217306](https://forums.developer.nvidia.com/t/xrdp-login-profile-different-from-boot-with-monitor/217306)). Crash inmediato reportado en JetPack 4.6 ([#259902](https://forums.developer.nvidia.com/t/jetson-nano-headless-fails/259902)).
- **vino + GNOME headless**: requiere display físico al arrancar, sin él no inicia. Workaround: **HDMI dummy plug** (~1 USD en AliExpress, "4K HDMI emulator").
- **NoMachine + GNOME**: alto CPU (20-30% por core idle) reportado en [foro NVIDIA #77399](https://forums.developer.nvidia.com/t/jetson-nano-vnc-headless-connections/77399). Fix: instalar Xfce4 y apuntar NX al desktop Xfce ([NoMachine KB AR02R01074](https://kb.nomachine.com/AR02R01074)).

### 7.3 Recomendación: **NoMachine + Xfce4 + HDMI dummy plug**

```bash
# 1. Descargar NoMachine arm64 .deb (Nano)
wget "https://downloads.nomachine.com/download/?id=114" -O nomachine_arm64.deb
sudo dpkg -i nomachine_arm64.deb

# 2. Instalar Xfce4 (reduce overhead RAM ~50 MB y elimina parpadeos)
sudo apt install -y xfce4 xfce4-terminal

# 3. Apuntar NX a Xfce
sudo nano /usr/NX/etc/node.cfg
# Descomentar/agregar:
#    CommandStartGnome xfce4-session
sudo /usr/NX/bin/nxserver --restart

# 4. Habilitar autologin en LightDM/GDM3 para que el Xserver esté listo al boot
sudo nano /etc/gdm3/custom.conf
# AutomaticLoginEnable=True
# AutomaticLogin=<tu_usuario>
```

Cliente Windows: descargar "NoMachine Free Edition" de [nomachine.com/download](https://www.nomachine.com/download). Apuntar a IP de la Nano + autenticación con user/password del Nano.

### 7.4 Si OpenGL no es crítico (preview headless por OpenCV directo a archivo o stream)

**TigerVNC sobre sesión Xfce virtual** es válido y no requiere dummy HDMI:

```bash
sudo apt install -y tigervnc-standalone-server tigervnc-common xfce4
mkdir -p ~/.vnc
echo '#!/bin/bash' > ~/.vnc/xstartup
echo 'startxfce4 &' >> ~/.vnc/xstartup
chmod +x ~/.vnc/xstartup
vncserver :1 -geometry 1280x720 -depth 24 -localhost no
# Desde Win11: TigerVNC viewer apuntando a <IP_Nano>:5901
```

Sin aceleración GPU, pero suficiente para mostrar imágenes clasificadas guardadas a disco.

---

## 8. SSH Windows 11 → Jetson Nano

### 8.1 Verificar OpenSSH client en Win11

```powershell
Get-WindowsCapability -Online -Name OpenSSH.Client*
ssh -V
# Esperado: OpenSSH_for_Windows_8.6p1, LibreSSL 3.4.3 (o más nuevo)
```

Si no está: `Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0`.

### 8.2 Generar clave ed25519 en Windows

```powershell
ssh-keygen -t ed25519 -C "mitgar14@embebidos3"
# Guarda en C:\Users\mitgar14\.ssh\id_ed25519 (privada) + id_ed25519.pub
# Pasphrase opcional (vacío para autoconexión sin prompt)
```

### 8.3 Copia clave a la Nano (sin `ssh-copy-id` en Windows)

```powershell
# Opción cmd nativa:
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh <usuario>@<IP_NANO> "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"

# Opción PowerShell pura:
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub" | ssh <usuario>@<IP_NANO> "mkdir -p ~/.ssh && tee -a ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

### 8.4 `~/.ssh/config` en Windows

Crear `C:\Users\mitgar14\.ssh\config` (Windows usa el mismo formato que Linux):

```sshconfig
Host nano
    HostName 192.168.X.X
    User <tu_usuario>
    IdentityFile C:/Users/mitgar14/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3
    LocalForward 8888 localhost:8888

Host nano-tail
    HostName 100.64.X.X
    User <tu_usuario>
    IdentityFile C:/Users/mitgar14/.ssh/id_ed25519
    ServerAliveInterval 60
    LocalForward 8888 localhost:8888

Host nano-bastion
    HostName <CONTABO_IP>
    User <tu_usuario>
    Port 2222
    IdentityFile C:/Users/mitgar14/.ssh/id_ed25519
```

Uso:
```powershell
ssh nano           # LAN local
ssh nano-tail      # Via Tailscale tailnet IP (100.x)
ssh nano-bastion   # Via reverse SSH tunnel en Contabo:2222
```

El bloque `LocalForward 8888 localhost:8888` ya port-forwarea JupyterLab — luego abrir `http://localhost:8888` en el browser.

### 8.5 sshd en la Nano

JetPack 4.6.1 trae `openssh-server` instalado y habilitado por default. Verificar:

```bash
sudo systemctl status ssh
# Si no está activo:
sudo systemctl enable ssh && sudo systemctl start ssh
```

---

## 9. Acceso remoto vía VPC Contabo

### 9.1 Bug crítico WireGuard en kernel 4.9-tegra

El módulo de kernel WireGuard (DKMS) **NO compila/funciona correctamente** en el kernel 4.9-tegra de JetPack 4.6.1. Error documentado en [foro NVIDIA #184764](https://forums.developer.nvidia.com/t/kernel-error-when-using-wireguard/184764):

```
Internal error: Accessing user space memory outside uaccess.h routines: 96000005
```

**Implicación:** NO ejecutar `apt install wireguard` esperando que el módulo cargue. Usar siempre **implementación userspace** (`wireguard-go` o cliente Tailscale, que usa `wireguard-go` internamente).

### 9.2 Alternativa A — Reverse SSH + autossh (simple, robusto)

**Concepto:** Nano abre una conexión persistente saliente hacia Contabo y mapea su SSH local al puerto 2222 del bastion. Cliente Win11 conecta a `ContaboIP:2222` y aterriza en la Nano.

**Prerequisito en Contabo** (`/etc/ssh/sshd_config`):

```
GatewayPorts clientspecified
AllowTcpForwarding yes
```

**Comando manual de prueba (en Nano):**

```bash
sudo apt install -y autossh
autossh -M 0 -fNR 0.0.0.0:2222:localhost:22 \
  -o "ServerAliveInterval=60" \
  -o "ServerAliveCountMax=3" \
  -o "ExitOnForwardFailure=yes" \
  -o "StrictHostKeyChecking=no" \
  -i /home/<usuario>/.ssh/id_ed25519 \
  <usuario_contabo>@<CONTABO_IP>
```

**Systemd unit persistente** (`/etc/systemd/system/autossh-tunnel.service`):

```ini
[Unit]
Description=AutoSSH reverse tunnel a Contabo
After=network-online.target
Wants=network-online.target

[Service]
User=<tu_usuario>
Environment="AUTOSSH_GATETIME=0"
Environment="AUTOSSH_POLL=60"
Environment="AUTOSSH_LOGFILE=/var/log/autossh.log"
ExecStart=/usr/bin/autossh -M 0 -NT \
  -o "ServerAliveInterval=60" \
  -o "ServerAliveCountMax=3" \
  -o "ExitOnForwardFailure=yes" \
  -o "StrictHostKeyChecking=no" \
  -R 0.0.0.0:2222:localhost:22 \
  -i /home/<tu_usuario>/.ssh/id_ed25519 \
  <usuario_contabo>@<CONTABO_IP>
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable autossh-tunnel
sudo systemctl start autossh-tunnel
sudo systemctl status autossh-tunnel  # verificar
```

Desde Win11: `ssh nano-bastion` (ver §8.4).

### 9.3 Alternativa B — WireGuard self-host (con `wireguard-go`)

Más complejo por el bug del kernel, pero viable. Setup completo:

**En Contabo (kernel moderno, WireGuard kernel-module funciona normal):**

```bash
sudo apt install -y wireguard
cd /etc/wireguard
wg genkey | tee server_private.key | wg pubkey > server_public.key
wg genkey | tee nano_private.key   | wg pubkey > nano_public.key
wg genkey | tee win11_private.key  | wg pubkey > win11_public.key
chmod 600 *_private.key

echo "net.ipv4.ip_forward=1" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

`/etc/wireguard/wg0.conf` (Contabo):

```ini
[Interface]
PrivateKey = <SERVER_PRIVATE>
Address = 10.42.0.1/24
ListenPort = 51820
PostUp = iptables -A FORWARD -i wg0 -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostDown = iptables -D FORWARD -i wg0 -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

[Peer]
PublicKey = <NANO_PUBLIC>
AllowedIPs = 10.42.0.2/32

[Peer]
PublicKey = <WIN11_PUBLIC>
AllowedIPs = 10.42.0.3/32
```

```bash
sudo systemctl enable wg-quick@wg0
sudo systemctl start wg-quick@wg0
sudo ufw allow 51820/udp
```

**En la Nano**: usar `wireguard-go` (userspace) en lugar del módulo:

```bash
# Instalar wireguard-tools y go
sudo apt install -y wireguard-tools golang-go

# Compilar wireguard-go
git clone https://git.zx2c4.com/wireguard-go
cd wireguard-go
make
sudo cp wireguard-go /usr/local/bin/

# Levantar interfaz wg0 con userspace driver
sudo WG_QUICK_USERSPACE_IMPLEMENTATION=wireguard-go wg-quick up wg0
```

`/etc/wireguard/wg0.conf` (Nano):

```ini
[Interface]
PrivateKey = <NANO_PRIVATE>
Address = 10.42.0.2/32

[Peer]
PublicKey = <SERVER_PUBLIC>
Endpoint = <CONTABO_IP>:51820
AllowedIPs = 10.42.0.0/24
PersistentKeepalive = 25
```

**En Win11**: app oficial WireGuard de [wireguard.com/install/](https://www.wireguard.com/install/) (firmado ZX2C4 LLC). Importar config:

```ini
[Interface]
PrivateKey = <WIN11_PRIVATE>
Address = 10.42.0.3/32
DNS = 1.1.1.1

[Peer]
PublicKey = <SERVER_PUBLIC>
Endpoint = <CONTABO_IP>:51820
AllowedIPs = 10.42.0.0/24
PersistentKeepalive = 25
```

Conexión Win11 → Nano: `ssh <usuario>@10.42.0.2`.

### 9.4 Alternativa C — Tailscale / Headscale (RECOMENDADA)

**Tailscale en la Nano (Ubuntu 18.04 bionic arm64 oficialmente soportado):**

```bash
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/bionic.noarmor.gpg | sudo tee /usr/share/keyrings/tailscale-archive-keyring.gpg >/dev/null
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/bionic.tailscale-keyring.list | sudo tee /etc/apt/sources.list.d/tailscale.list

sudo apt update
sudo apt install -y tailscale
sudo systemctl enable tailscaled
sudo tailscale up
# Imprime URL https://login.tailscale.com/a/XXXXX — abrir en browser, autenticar con Google/MS/GitHub
```

Tras `tailscale up`, la Nano obtiene una IP `100.64.X.X` (CGNAT range) accesible desde cualquier dispositivo en el mismo tailnet.

**Tailscale en Win11:**

Descargar instalador `.exe` de [tailscale.com/download/windows](https://tailscale.com/download/windows). Login con la misma cuenta. La Win11 obtiene su propia IP `100.64.X.Y`.

Conexión: `ssh nano-tail` (ver §8.4) que apunta a `100.64.X.X`.

**Tailscale SSH (opcional, simplifica más)** — gestiona auth SSH a través de Tailscale, sin necesidad de gestionar claves manualmente:

```bash
# En la Nano:
sudo tailscale up --ssh
# Ahora desde Win11: ssh <usuario>@100.64.X.X funciona sin clave configurada
```

Ver [tailscale.com/kb/1193/tailscale-ssh](https://tailscale.com/kb/1193/tailscale-ssh).

**Headscale (control plane self-hosted en Contabo)** — sólo si quieres autonomía total de Tailscale Inc.:

```bash
# En Contabo:
sudo apt install -y headscale
sudo mkdir -p /etc/headscale /var/lib/headscale
sudo nano /etc/headscale/config.yaml
# Configurar server_url, listen_addr, base_domain, etc.
sudo systemctl enable headscale
sudo systemctl start headscale

# Crear namespace y pre-auth key:
sudo headscale users create embebidos3
sudo headscale preauthkeys create --user embebidos3 --expiration 24h --reusable
```

En la Nano:

```bash
sudo tailscale down
sudo tailscale up --login-server=https://<CONTABO_DOMAIN>:8080 --authkey=<PREAUTH_KEY>
```

Requiere dominio con HTTPS en Contabo (Let's Encrypt + Caddy/Nginx reverse proxy). Setup completo en [headscale.net](https://headscale.net).

### 9.5 Tabla comparativa alternativas A / B / C

| Criterio | A: Reverse SSH | B: WireGuard wireguard-go | **C: Tailscale free** | C': Headscale |
|---|---|---|---|---|
| Complejidad setup | Baja (1 unit file) | Media-alta (compilar wireguard-go) | **Muy baja** (`apt install` + login) | Media (dominio HTTPS + config) |
| Performance | Buena (TCP SSH) | Óptima (UDP, kernel en hub) | Buena (UDP, P2P cuando posible) | Igual que Tailscale |
| Dependencia Tailscale Inc. | Ninguna | Ninguna | **Sí** (control plane + DERP fallback) | Ninguna |
| NAT traversal automático | No | No | **Sí** | Sí (con tu propio DERP server) |
| Conexión P2P directa | No (Contabo siempre relay) | No (Contabo siempre hub) | **Sí** cuando red lo permite | Sí |
| Resilencia a falla Contabo | Sin acceso | Sin acceso | Tailscale Inc. + DERP siguen | Sin acceso (Headscale en Contabo) |
| Mejor para | Backup simple, scripting | Autonomía sin dependencia | **Day-to-day** | Producción seria self-hosted |

**Veredicto:** **Tailscale free tier** como primera línea. **Reverse SSH autossh** como fallback. Headscale opcional si se quiere control total más adelante.

---

## 10. Setup VPC Contabo como bastion

### 10.1 Inicialización segura

```bash
# Como root vía SSH inicial (recibido en email Contabo):
apt update && apt upgrade -y

# Crear usuario deploy con sudo:
adduser deploy
usermod -aG sudo deploy

# Pasar authorized_keys de root al nuevo user:
mkdir -p /home/deploy/.ssh
cp /root/.ssh/authorized_keys /home/deploy/.ssh/
chown -R deploy:deploy /home/deploy/.ssh
chmod 700 /home/deploy/.ssh
chmod 600 /home/deploy/.ssh/authorized_keys
```

### 10.2 Hardening sshd (`/etc/ssh/sshd_config`)

```sshconfig
Port 2200
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
AllowUsers deploy
X11Forwarding no
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2
GatewayPorts clientspecified
AllowTcpForwarding yes
```

```bash
# CRÍTICO: probar conexión nueva ANTES de cerrar la actual
sudo systemctl restart sshd
# Desde otra terminal: ssh -p 2200 deploy@<CONTABO_IP>
```

### 10.3 UFW firewall

```bash
sudo apt install -y ufw
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 2200/tcp          # SSH custom
sudo ufw allow 2222/tcp          # reverse SSH tunnel
sudo ufw allow 51820/udp         # WireGuard (si alternativa B)
sudo ufw allow 80/tcp            # HTTP (si Headscale)
sudo ufw allow 443/tcp           # HTTPS (si Headscale)
sudo ufw enable
sudo ufw status verbose
```

### 10.4 fail2ban opcional

```bash
sudo apt install -y fail2ban
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
sudo nano /etc/fail2ban/jail.local
# [sshd] enabled = true, port = 2200
sudo systemctl restart fail2ban
```

---

## 11. Topología final demo (3 escenarios)

### Escenario A — WiFi UAO disponible

```
                    Internet (UAO WiFi)
                          │
   ┌──────────────────────┼──────────────────────┐
   │                      │                      │
Jetson Nano          Contabo VPC            Win11 Laptop
(192.168.x.x)        (backup only,         (DHCP UAO)
tailscale:           con Headscale)        tailscale:
100.64.X.2                                 100.64.X.3

Protocolo: Tailscale P2P directo (misma red → latencia mínima)
Alternativa: SSH directo a IP local Nano
Comando: ssh nano  (config con IP local)
        ssh nano-tail (config con IP tailscale)
```

### Escenario B — Sin WiFi UAO, hotspot móvil

```
                  Hotspot móvil (LTE/5G)
                          │
   ┌──────────────────────┼──────────────────────┐
   │                      │                      │
Jetson Nano          Contabo VPC            Win11 Laptop
(NAT del hotspot)    (DERP relay si        (NAT del hotspot)
tailscale:           NAT estricto)         tailscale:
100.64.X.2                                 100.64.X.3

Protocolo: Tailscale (P2P primero; DERP fallback si NAT estricto)
Fallback: Reverse SSH via Contabo:2222 → autossh siempre activo
Comando: ssh nano-tail  (recomendado)
         ssh nano-bastion  (fallback)
```

### Escenario C — Nano en modo AP, laptop directo (sin Internet)

```
Win11 Laptop ─── WiFi direct ─── Jetson Nano (AP)
(DHCP del AP                     192.168.42.1 (wlan1)
 192.168.42.x)                   hostapd + dnsmasq

Sin Internet → sin Tailscale, sin Contabo
Protocolo: SSH directo a 192.168.42.1
NoMachine: cliente apuntando a 192.168.42.1
Comando: ssh usuario@192.168.42.1
```

**Setup Escenario C** (Nano se convierte en AP usando TL-WN722N v4):

```bash
# Una vez instalado el driver out-of-tree (§4) y blacklisted r8188eu (§4.4):
sudo nmcli con up embebidos3-hotspot ifname wlan1
# O con hostapd manual (§5)
```

---

## 12. Overhead de overlay networks en Cortex-A57 — evidencia académica

### 12.1 Disclaimer sobre el estado de la evidencia

**No existe paper peer-reviewed que mida CPU overhead, latencia p50/p99 ni battery drain de Tailscale/ZeroTier/WireGuard específicamente sobre Cortex-A57 (Jetson Nano) o A72 (RPi 4) en condiciones de campo.** Lo que sigue se basa en (a) benchmarks comunitarios reproducibles, (b) un reporte UvA en x86 que da el overhead relativo kernel-vs-userspace, (c) extrapolación arquitectural A72→A57.

### 12.2 Throughput WireGuard puro en ARM (datos `cyyself/wg-bench` loopback netns)

Mediciones del repo [cyyself/wg-bench](https://github.com/cyyself/wg-bench) (Shell, 226 ⭐, abril 2026):

| Dispositivo | CPU / Frecuencia | OS / Kernel | Throughput |
|---|---|---|---|
| Raspberry Pi 4 | Cortex-A72 1,50 GHz | OpenWRT 23 / Linux 5.15 | **881 Mbps** |
| Raspberry Pi 4 | Cortex-A72 1,80 GHz stock | RPi OS trixie / Linux 6.12 | **777 Mbps** |
| Raspberry Pi 4 | Cortex-A72 2,00 GHz OC | OpenWRT 23 / Linux 5.15 | **1,02 Gbps** |
| Raspberry Pi 3B | Cortex-A53 1,20 GHz | OpenWRT 23 / Linux 5.15 | 522 Mbps |

**Extrapolación al Jetson Nano (Cortex-A57 1,43 GHz):** A57 es arquitecturalmente similar al A72 pero con IPC ligeramente inferior. Estimación conservadora: **600-750 Mbps** de throughput WireGuard kernel en loopback. **NO medido directamente** — sólo extrapolado. Estos números son el techo CPU del cifrado, no el throughput sobre WiFi real.

**El cuello de botella real en este proyecto es WiFi 802.11g a ~20 Mbps efectivos** (TL-WN722N v4, ver §5.1), o WiFi UAO/móvil a ~50-100 Mbps. A 100 Mbps, **el cifrado consume <11% de la capacidad CPU** disponible del Nano. Despreciable en términos de FPS de inferencia.

### 12.3 WireGuard kernel vs WireGuard-Go (Tailscale) — overhead CPU

Del reporte [Dekker & Spaans (UvA, 2020)](https://rp.os3.nl/2019-2020/p71/report.pdf) en hardware x86 1 Gbps:

| Implementación | Goodput TCP | CPU (1 core) | Latencia mediana |
|---|---|---|---|
| WireGuard kernel | ~940 Mbps | ~45 % | bajo |
| **WireGuard-Go (userspace) — usado por Tailscale en Linux** | ~540 Mbps | **~85 %** | el más alto |
| strongSwan AES-GCM | ~950 Mbps | ~40 % | el más bajo |
| OpenVPN AES-256-CBC | ~200 Mbps | ~75 % | alto |

**Conclusión transferible a ARM:** la relación kernel-vs-userspace (~1,5-2× más CPU para userspace) es válida arquitecturalmente. En el Nano a 100 Mbps:

- WireGuard kernel ideal: ~3-5 % CPU de un núcleo.
- WireGuard-Go (Tailscale): ~6-10 % CPU de un núcleo.

**Ambos despreciables para el proyecto.** Track B inferencia YOLOv8n TRT corre en GPU Maxwell (no compite por CPU); Track A SSD TFLite corre en CPU con XNNPACK pero deja 3 núcleos para overhead de red.

### 12.4 RAM footprint medido (datos reales GitHub issues + dev.to)

| Solución | RSS real ARM Linux |
|---|---|
| WireGuard kernel module | <5 MB (sólo herramientas userspace) |
| **Tailscale daemon `tailscaled`** | **~30-50 MB RSS** (binario 20 MB en disco) |
| ZeroTier daemon | ~30-50 MB RSS |

**Aclaración crítica:** valores `VIRT` ~540 MB que aparecen en algunos issues de Tailscale (ej. [#15435](https://github.com/tailscale/tailscale/issues/15435)) son **memoria virtual reservada por el runtime Go, no RSS físico**. El consumo físico real (`RES` en htop) son 30-50 MB.

Sobre los 4 GB unificados de la Nano, 50 MB es 1,2 % — no compite con TRT engine build (~3,5 GB pico).

### 12.5 ZeroTier — falta de evidencia

**No se encontró ningún benchmark reproducible de ZeroTier en ARM Cortex-A57/A72.** El reporte [NetFoundry 2022](https://netfoundry.io/benchmark/benchmarking%20open%20source%20networking.pdf) compara throughput en x86 cloud (Phoenix-Ashburn): Tailscale ~58 Mbps vs WireGuard standalone ~36-45 Mbps vs ZeroTier ~18-23 Mbps, pero estos números reflejan **NAT traversal y routing**, no eficiencia criptográfica. Es propietario y sin revisión independiente. **No usar como base para decisión.**

### 12.6 OpenVPN — descartado para edge battery-powered

Throughput máximo ~200 Mbps en x86 con AES-256-CBC. En ARM Cortex-A57 (sin AES-NI hardware en ARMv8.0), el CPU overhead es significativamente mayor que WireGuard. **No recomendado para el proyecto.**

### 12.7 Verdict final

En el régimen relevante del proyecto (**100 Mbps WiFi**, **un cliente**, **demo de minutos a horas**):

| Métrica | Veredicto |
|---|---|
| CPU overhead WireGuard kernel | Despreciable (~3-5 % de un núcleo) |
| CPU overhead Tailscale (WireGuard-Go) | Despreciable (~6-10 % de un núcleo) |
| RAM Tailscale daemon | Insignificante (50 MB de 4 GB = 1,2 %) |
| Battery drain | **No medido en literatura** — extrapolación: marginal sobre el consumo total del Nano + inferencia |

**Recomendación reforzada de Tailscale como D17:** los criterios diferenciadores son no-CPU:

1. **NAT traversal automático** (Tailscale gana sobre WireGuard standalone).
2. **Footprint operacional** (Tailscale gana sobre setup manual de claves WireGuard).
3. **Autonomía de servicio externo** (WireGuard standalone gana; Headscale en Contabo cierra el gap).

### 12.8 Cómo cerrar el gap empírico en 30 minutos

Si se quiere dato real en la Nano antes del deploy:

```bash
# Setup en la Nano:
sudo apt install -y wireguard-tools iperf3
git clone https://github.com/cyyself/wg-bench
cd wg-bench
sudo ./setup-netns.sh
sudo ./benchmark.sh   # imprime Mbps reales en netns A57

# Medir Tailscale en paralelo:
sudo tailscale up
# Desde otro nodo del tailnet:
iperf3 -c 100.64.X.X -t 60
# En la Nano simultáneamente: htop → ver % CPU del proceso tailscaled
```

Reportar resultado en próxima ronda actualizable.

---

## 13. Recomendación final consolidada

### Stack a desplegar HOY (orden de instalación):

1. **SSH directo Win11 → Nano LAN** (§8) — funciona desde minuto 0 sin nada externo.
2. **Driver RTL8188EUS DKMS + blacklist** (§4) — habilita la antena TL-WN722N v4 con AP mode.
3. **Tailscale en Nano + Win11** (§9.4) — acceso desde cualquier red, prácticamente cero config.
4. **NoMachine + Xfce4 + dummy HDMI** (§7.3) — escritorio remoto con OpenGL accel para preview OpenCV.
5. **Contabo bastion endurecido + autossh systemd** (§10 + §9.2) — backup independiente de Tailscale Inc.
6. **NetworkManager hotspot perfil `embebidos3-hotspot`** (§6) — listo para escenario demo C sin Internet.

### Lo que NO se debe hacer

- ❌ `apt install wireguard` esperando que el módulo cargue en kernel 4.9-tegra (§9.1).
- ❌ xrdp en Jetson Nano JetPack 4.6.1 (§7.2).
- ❌ vino headless sin HDMI dummy plug (§7.2).
- ❌ Cargar `r8188eu` mainline sin blacklist cuando se quiere AP mode (§4.4).
- ❌ Configurar hostapd con WPA3/SAE/ieee80211w con RTL8188EUS (§5.1).

---

## 14. Comandos copy-paste end-to-end (ejecutar en orden, en la Nano)

```bash
#!/bin/bash
set -euo pipefail

# === 1. Preparación del sistema ===
sudo apt-get update
sudo apt-get install -y git build-essential dkms bc \
    linux-headers-$(uname -r) autossh \
    xfce4 xfce4-terminal hostapd dnsmasq

# === 2. Driver RTL8188EUS para TL-WN722N v4 ===
cd /tmp
git clone https://github.com/aircrack-ng/rtl8188eus.git -b v5.3.9
cd rtl8188eus
sudo bash dkms-install.sh

# === 3. Blacklist módulos conflicting ===
sudo tee /etc/modprobe.d/realtek.conf <<'EOF'
blacklist r8188eu
blacklist rtl8xxxu
EOF
sudo tee /etc/modprobe.d/8188eu.conf <<'EOF'
options 8188eu rtw_power_mgnt=0 rtw_enusbss=0 rtw_hwpwrp_detect=0 rtw_low_power=0 rtw_ips_mode=1
EOF
sudo update-initramfs -u

# === 4. NoMachine ===
cd /tmp
wget "https://downloads.nomachine.com/download/?id=114" -O nomachine_arm64.deb
sudo dpkg -i nomachine_arm64.deb || sudo apt-get install -f -y
# Apuntar a Xfce — editar /usr/NX/etc/node.cfg y descomentar:
# CommandStartGnome xfce4-session
sudo /usr/NX/bin/nxserver --restart

# === 5. Tailscale ===
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/bionic.noarmor.gpg | \
    sudo tee /usr/share/keyrings/tailscale-archive-keyring.gpg >/dev/null
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/bionic.tailscale-keyring.list | \
    sudo tee /etc/apt/sources.list.d/tailscale.list
sudo apt update
sudo apt install -y tailscale
sudo systemctl enable tailscaled
sudo tailscale up --ssh
# Seguir URL impresa para autenticar

# === 6. Hotspot NetworkManager (escenario C) ===
sudo nmcli con add type wifi ifname wlan1 con-name embebidos3-hotspot \
    autoconnect no ssid "embebidos3-nano"
sudo nmcli con modify embebidos3-hotspot \
    802-11-wireless.mode ap \
    802-11-wireless.band bg \
    802-11-wireless.channel 6 \
    802-11-wireless-security.key-mgmt wpa-psk \
    802-11-wireless-security.psk "<TU_PASSWORD_WPA2>" \
    ipv4.method shared \
    ipv4.addresses 192.168.42.1/24
# Activar manualmente cuando se necesite:
# sudo nmcli con up embebidos3-hotspot ifname wlan1

# === 7. autossh reverse tunnel a Contabo ===
sudo tee /etc/systemd/system/autossh-tunnel.service <<EOF
[Unit]
Description=AutoSSH reverse tunnel a Contabo
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
Environment="AUTOSSH_GATETIME=0"
Environment="AUTOSSH_POLL=60"
Environment="AUTOSSH_LOGFILE=/var/log/autossh.log"
ExecStart=/usr/bin/autossh -M 0 -NT \\
  -o "ServerAliveInterval=60" \\
  -o "ServerAliveCountMax=3" \\
  -o "ExitOnForwardFailure=yes" \\
  -o "StrictHostKeyChecking=no" \\
  -R 0.0.0.0:2222:localhost:22 \\
  -i /home/$USER/.ssh/id_ed25519 \\
  <USUARIO_CONTABO>@<CONTABO_IP>
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable autossh-tunnel

echo "Reboot requerido para aplicar blacklist módulos kernel."
echo "Tras reboot, verificar: lsmod | grep 8188eu (sin la r); iw list | grep AP"
```

---

## 15. Decisiones nuevas (extiende ledger D1-D15)

| # | Decisión | Justificación | Fuente |
|---|---|---|---|
| **D16** | Cliente Wi-Fi USB en Nano: TP-Link TL-WN722N v4 + driver `aircrack-ng/rtl8188eus@v5.3.9` instalado vía DKMS, con `r8188eu` mainline blacklisteado | El v4 usa RTL8188EUS Realtek (no AR9271 del v1). Sólo el driver out-of-tree soporta AP mode. Sin blacklist el kernel carga el módulo staging que no soporta AP. | §3, §4 + issue [aircrack-ng/rtl8188eus#50](https://github.com/aircrack-ng/rtl8188eus/issues/50) |
| **D17** | Acceso remoto Win11 ↔ Nano: SSH OpenSSH ed25519 + **Tailscale free tier** como overlay primario. **Reverse SSH autossh systemd → Contabo:2222** como fallback. | Tailscale evita el bug kernel 4.9-tegra de WireGuard (usa wireguard-go userspace), tiene NAT traversal automático, y permite P2P directo cuando red lo permite. Reverse SSH como respaldo independiente de Tailscale Inc. | §9.1, §9.4 + foro [NVIDIA #184764](https://forums.developer.nvidia.com/t/kernel-error-when-using-wireguard/184764) |
| **D18** | Escritorio remoto: **NoMachine + Xfce4 + HDMI dummy plug**. xrdp y vino descartados. | Único stack con OpenGL accel (para preview OpenCV en vivo), bajo overhead red (NX), cliente Win11 oficial, autostart confiable. xrdp tiene EGL roto en Nano confirmado por moderadores NVIDIA. vino requiere display físico. | §7 + foros [#77399](https://forums.developer.nvidia.com/t/jetson-nano-vnc-headless-connections/77399), [#217306](https://forums.developer.nvidia.com/t/xrdp-login-profile-different-from-boot-with-monitor/217306) |
| **D19** | Demo final: dos rutas de conectividad probadas — (a) hotspot móvil + Tailscale (preferida), (b) Nano como AP standalone con `embebidos3-hotspot` NetworkManager + Win11 conectado directo (fallback sin Internet). | El UAO no garantiza WiFi conocida. Hotspot móvil con Tailscale resuelve si hay LTE. AP mode resuelve sin Internet. AP usa RTL8188EUS con config WPA2-PSK 2,4 GHz minimal del issue #261. | §5.1, §6, §11 escenarios B y C |
| **D20** | VPC Contabo configurada como bastion: sshd hardened en puerto 2200 + `GatewayPorts clientspecified` + UFW + opcionalmente Headscale self-hosted. | Bastion independiente de Tailscale Inc. permite reverse SSH persistente. Headscale en Contabo da control total si en el futuro se quiere autonomía. | §10 |

Estas decisiones se incorporan al [`decisiones-D1-D15-ledger.md`](decisiones-D1-D15-ledger.md) en su próxima actualización.

---

## 16. Fuentes consultadas (acumuladas Ronda 6)

| # | Título | URL | Tipo | Relevancia |
|---|---|---|---|---|
| 1 | aircrack-ng/rtl8188eus (driver canónico) | https://github.com/aircrack-ng/rtl8188eus | Repo | Driver elegido |
| 2 | aircrack-ng/rtl8188eus issue #50 (compilación arm64 k4.9) | https://github.com/aircrack-ng/rtl8188eus/issues/50 | Issue | Confirma kernel 4.9 aarch64 |
| 3 | aircrack-ng/rtl8188eus issue #261 (AP mode hostapd config) | https://github.com/aircrack-ng/rtl8188eus/issues/261 | Issue | Config WPA2-PSK minimal |
| 4 | lwfinger/rtl8188eu (alternativa standalone) | https://github.com/lwfinger/rtl8188eu | Repo | Driver alternativo |
| 5 | morrownr/8821cu issue #129 (Jetson Nano k4.9.337-tegra) | https://github.com/morrownr/8821cu-20210916/issues/129 | Issue | Confirma kernel exacto |
| 6 | lakinduakash/linux-wifi-hotspot | https://github.com/lakinduakash/linux-wifi-hotspot | Repo | Sucesor create_ap |
| 7 | Foro NVIDIA #77399 — Jetson Nano VNC headless | https://forums.developer.nvidia.com/t/jetson-nano-vnc-headless-connections/77399 | Foro | NoMachine CPU overhead |
| 8 | Foro NVIDIA #107552 — Headless VNC sin monitor | https://forums.developer.nvidia.com/t/headless-vnc-access-without-attached-monitor/107552 | Foro | HDMI dummy plug solución |
| 9 | Foro NVIDIA #217306 — xrdp EGL roto | https://forums.developer.nvidia.com/t/xrdp-login-profile-different-from-boot-with-monitor/217306 | Foro | xrdp descartado |
| 10 | Foro NVIDIA #259902 — xrdp crash JetPack 4.6 | https://forums.developer.nvidia.com/t/jetson-nano-headless-fails/259902 | Foro | Confirmación crash |
| 11 | Foro NVIDIA #184764 — WireGuard kernel error tegra | https://forums.developer.nvidia.com/t/kernel-error-when-using-wireguard/184764 | Foro | Bug crítico k4.9-tegra |
| 12 | Foro NVIDIA #154317 — Remote access over internet | https://forums.developer.nvidia.com/t/remote-access-to-jetson-over-the-internet/154317 | Foro | NoMachine ARM dummy HDMI |
| 13 | NoMachine KB AR02R01074 (tips Jetson Nano) | https://kb.nomachine.com/AR02R01074 | Doc oficial | Xfce4 recomendado |
| 14 | JetsonHacks NoMachine guide | https://jetsonhacks.com/2023/12/03/nomachine-jetson-remote-desktop/ | Blog | Walkthrough |
| 15 | JetsonHacks Wi-Fi Hotspot Setup | (video) https://youtu.be/GAOvGAdwiHk | Video | nmcli hotspot |
| 16 | JetsonHacks NoMachine - Jetson Remote Desktop on Windows | (video) https://youtu.be/OYrSADrtSag | Video | Setup Win11 |
| 17 | couka.de — TigerVNC + Xfce JetPack 4.4 | https://couka.de/2020/10/26/jetson-nano-enabling-headless-vnc-connection-on-jetpack-4-4-incl-installing-xfce/ | Blog | TigerVNC walkthrough |
| 18 | overclock98/Jetson_Nano_true_Headless_setup | https://github.com/overclock98/Jetson_Nano_true_Headless_setup_without_hdmi_display | Repo | Headless completo |
| 19 | amirulhakimizaini23/Jetson-Nano-Remote-Desktop | https://github.com/amirulhakimizaini23/Jetson-Nano-Remote-Desktop | Repo | Tailscale+NoMachine combo |
| 20 | JetsonHacksNano/buildKernelAndModules | https://github.com/JetsonHacksNano/buildKernelAndModules | Repo | Build kernel modules en Nano |
| 21 | Tailscale docs — install Linux | https://tailscale.com/docs/install/linux | Doc oficial | Ubuntu bionic arm64 soportado |
| 22 | Tailscale docs — SSH feature | https://tailscale.com/docs/features/tailscale-ssh | Doc oficial | --ssh flag |
| 23 | Tailscale vs WireGuard | https://tailscale.com/compare/wireguard | Doc oficial | Comparativa |
| 24 | pkgs.tailscale.com Ubuntu bionic | https://pkgs.tailscale.com/stable/ubuntu/ | Doc oficial | Repo APT |
| 25 | Headscale (juanfont/headscale) | https://github.com/juanfont/headscale | Repo | Self-hosted control plane |
| 26 | Headscale.net (docs) | https://headscale.net | Doc oficial | Setup completo |
| 27 | WireGuard install (Win11 cliente) | https://www.wireguard.com/install/ | Doc oficial | Cliente Windows firmado |
| 28 | Pro Custodibus — WireGuard Hub-and-Spoke | https://procustodibus.com/blog/2020/11/wireguard-hub-and-spoke-config/ | Tutorial | Topología hub WG |
| 29 | Kyle Manna — SSH reverse tunnel systemd | https://blog.kylemanna.com/linux/ssh-reverse-tunnel-on-linux-with-systemd | Blog | autossh systemd unit |
| 30 | AskUbuntu #1316798 — persistent autossh | https://askubuntu.com/questions/1316798 | StackExchange | Unit file ejemplo |
| 31 | SuperUser #1747549 — ssh-copy-id en Windows | https://superuser.com/questions/1747549 | StackExchange | type / Get-Content workaround |
| 32 | Microsoft Learn — OpenSSH Windows install | https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse | Doc oficial | OpenSSH client Win11 |
| 33 | nvidia-jetson.piveral.com — Jetpack 6 RTL8188EUS | https://nvidia-jetson.piveral.com/jetson-orin-nano/jetpack-6-doesnt-support-rtl8188eus | KB | Trends JP6 (informativo) |
| 34 | selfhosting.sh — Remote Access comparativa | https://selfhosting.sh/foundations/remote-access | Comparativa | Tabla Tailscale/WG/CF Tunnel/SSH |
| 35 | Code With Aarohi — Jetson Nano Headless tutorial | (video) https://youtu.be/7-WMvmWVxJQ | Video | Setup WiFi + headless |
| 36 | 2GuysTek — Ditched VPN for Tailscale | (video) https://youtu.be/yGWVYGUU6Pg | Video | Tailscale uso real |
| 37 | Tailscale — Windows + Remote Desktop | (video) https://youtu.be/dVCOY_Z-5bs | Video | Win11 install walkthrough |
| 38 | Lawrence Systems — Pi + ZeroTier | (video) https://youtu.be/L3KIhZxvQ5A | Video | ZeroTier comparación |
| 39 | Dekker & Spaans (UvA 2020) — VPN performance comparison | https://rp.os3.nl/2019-2020/p71/report.pdf | Reporte académico | WireGuard-C vs Go vs OpenVPN |
| 40 | Anbarje & Sabbagh (Linnaeus 2020) — WireGuard vs OpenVPN | https://www.diva-portal.org/smash/get/diva2:1467354/FULLTEXT01.pdf | Tesis | Baseline OpenVPN |
| 41 | cyyself/wg-bench (benchmark comunitario WireGuard ARM) | https://github.com/cyyself/wg-bench | Repo | Throughput RPi 4 reproducible |
| 42 | Pro Custodibus — WireGuard Performance Tuning | https://www.procustodibus.com/blog/2022/12/wireguard-performance-tuning/ | Blog técnico | CPU usage x86 ref |
| 43 | NetFoundry 2022 — Benchmark open source networking | https://netfoundry.io/benchmark/benchmarking%20open%20source%20networking.pdf | Reporte (propietario) | Comparativa, baja confianza |
| 44 | Tailscale issue #15435 — RAM footprint VIRT vs RSS | https://github.com/tailscale/tailscale/issues/15435 | Issue | Aclaración Go runtime VIRT |

---

## 17. Anexo: Citas timestamp de videos YouTube relevantes

### JetsonHacks — Wi-Fi Hotspot Setup (video `GAOvGAdwiHk`, 8 min)

Capítulos cronológicos:

- `0:00-0:20` — Intro
- `0:20-0:54` — Network Manager Automatic Setup ([https://youtu.be/GAOvGAdwiHk?t=20](https://youtu.be/GAOvGAdwiHk?t=20))
- `0:54-2:53` — Configuration Workaround ([https://youtu.be/GAOvGAdwiHk?t=54](https://youtu.be/GAOvGAdwiHk?t=54))
- `2:53-3:40` — Jetson as Wi-Fi Router ([https://youtu.be/GAOvGAdwiHk?t=173](https://youtu.be/GAOvGAdwiHk?t=173))
- `3:40-5:25` — Network Manager Manual Configuration ([https://youtu.be/GAOvGAdwiHk?t=220](https://youtu.be/GAOvGAdwiHk?t=220))
- `5:25-6:12` — Connect to Jetson Wirelessly via SSH ([https://youtu.be/GAOvGAdwiHk?t=325](https://youtu.be/GAOvGAdwiHk?t=325))
- `6:12-8:13` — CLI wireless tools overview ([https://youtu.be/GAOvGAdwiHk?t=372](https://youtu.be/GAOvGAdwiHk?t=372))

### JetsonHacks — NoMachine Jetson Remote Desktop on Windows (video `OYrSADrtSag`, 9 min)

- `0:00-0:26` — Intro
- `0:26-1:27` — Installing NoMachine ([https://youtu.be/OYrSADrtSag?t=26](https://youtu.be/OYrSADrtSag?t=26))
- `1:27-2:23` — Installing on Windows ([https://youtu.be/OYrSADrtSag?t=87](https://youtu.be/OYrSADrtSag?t=87))
- `2:23-7:21` — Using NoMachine ([https://youtu.be/OYrSADrtSag?t=143](https://youtu.be/OYrSADrtSag?t=143))
- `7:21-8:34` — Testing NoMachine ([https://youtu.be/OYrSADrtSag?t=441](https://youtu.be/OYrSADrtSag?t=441))
- `8:34-9:11` — Conclusion ([https://youtu.be/OYrSADrtSag?t=514](https://youtu.be/OYrSADrtSag?t=514))

### 2GuysTek — Ditched VPN for Tailscale (video `yGWVYGUU6Pg`, 8 min)

- `0:00-1:11` — Intro
- `1:11-1:53` — A background on Tailscale ([https://youtu.be/yGWVYGUU6Pg?t=71](https://youtu.be/yGWVYGUU6Pg?t=71))
- `1:53-3:43` — Pre-Tailscale remote access pattern ([https://youtu.be/yGWVYGUU6Pg?t=113](https://youtu.be/yGWVYGUU6Pg?t=113))
- `3:43-5:21` — Tailscale homelab pattern ([https://youtu.be/yGWVYGUU6Pg?t=223](https://youtu.be/yGWVYGUU6Pg?t=223))
- `5:21-6:16` — IPSec vs Tailscale performance ([https://youtu.be/yGWVYGUU6Pg?t=321](https://youtu.be/yGWVYGUU6Pg?t=321))
- `6:16-7:00` — Pricing + Zero Trust ([https://youtu.be/yGWVYGUU6Pg?t=376](https://youtu.be/yGWVYGUU6Pg?t=376))
- `7:00-7:46` — Closing ([https://youtu.be/yGWVYGUU6Pg?t=420](https://youtu.be/yGWVYGUU6Pg?t=420))

### Lawrence Systems — Raspberry Pi + ZeroTier (video `L3KIhZxvQ5A`, 16 min)

- `0:00-1:30` — Intro
- `4:15-6:45` — ZeroTier explained ([https://youtu.be/L3KIhZxvQ5A?t=255](https://youtu.be/L3KIhZxvQ5A?t=255))
- `6:45-8:20` — Installation ([https://youtu.be/L3KIhZxvQ5A?t=405](https://youtu.be/L3KIhZxvQ5A?t=405))
- `8:20-13:45` — Installing ZeroTier ([https://youtu.be/L3KIhZxvQ5A?t=500](https://youtu.be/L3KIhZxvQ5A?t=500))

### Tailscale official — Windows + RDP (video `dVCOY_Z-5bs`, 13 min)

- `0:00-0:51` — Start
- `0:51-9:10` — Install Tailscale on Windows ([https://youtu.be/dVCOY_Z-5bs?t=51](https://youtu.be/dVCOY_Z-5bs?t=51))
- `9:10-12:54` — Remote Desktop Quickstart ([https://youtu.be/dVCOY_Z-5bs?t=550](https://youtu.be/dVCOY_Z-5bs?t=550))

---

## 18. Historial de investigación

| Ronda | Fecha | Profundidad | Foco principal |
|---|---|---|---|
| 6 (esta) | 2026-05-12 | Alto | Acceso remoto Win11 → Nano + TL-WN722N v4 + Contabo bastion + overlay networking |

---

## 19. Gaps de evidencia residuales

1. **Compilación RTL8188EUS específicamente en JetPack 4.6.1 (kernel 4.9.337-tegra)** sin testimonio directo. Inferencia: el issue #50 documenta build en `4.9.223-arm64`, estructura idéntica. Mitigación: probar build + reportar resultado en la próxima ronda.
2. **NoMachine en Xfce4 — overhead RAM exacto en Nano 4 GB**: estimación 40-50 MB; medir empíricamente tras instalación.
3. **Tailscale P2P vs DERP en red UAO**: depende del NAT del campus. Estimación pesimista: fallback a DERP, latencia +50-150 ms. Verificable sólo en sitio.
4. **Throughput máximo TL-WN722N v4 en modo AP con varios clientes**: estimación 15-20 Mbps reales; para demo (1 cliente Win11) sobra.
5. **Drift batería en MAXN con WiFi+inferencia activos**: independiente de esta ronda; ver investigación previa sobre power management.

---

## 20. Cross-references

- [`decisiones-D1-D15-ledger.md`](decisiones-D1-D15-ledger.md) — actualizar para incorporar D16-D20.
- [`HANDOFF-implementacion-vastai-hf.md`](HANDOFF-implementacion-vastai-hf.md) §5 — los notebooks pueden aprovechar el túnel SSH + port-forward para JupyterLab local en la Nano si en algún momento se mueve workflow a Nano.
- [`compatibilidad-stack-cloud-jetson.md`](compatibilidad-stack-cloud-jetson.md) — el stack de drivers RTL8188EUS NO interfiere con el stack de inferencia TFLite/TRT del deploy final.

---

**Fin de la Ronda 6.** Cualquier cambio sustancial al stack de acceso remoto requiere nueva ronda `/investiga`.

---

## 21. Ronda 7 — fix DKMS install RTL8188EUS en JetPack 4.6.x (2026-05-12, profundidad baja)

### 21.1 Motivación

Tras ejecutar `sudo bash dkms-install.sh` del repo `aircrack-ng/rtl8188eus@v5.3.9` siguiendo §14, el comando devolvió:

```
DKMS: add completed.
Error! Your kernel headers for kernel 4.9.253-tegra cannot be found.
Please install the linux-headers-4.9.253-tegra package,
or use the --kernelsourcedir option to tell DKMS where it's located
```

Esto reveló **3 correcciones necesarias** al doc original.

### 21.2 Tabla definitiva kernel ↔ JetPack 4.6.x (verificada R7 + corrección post-confirmación)

**Confirmado empíricamente 2026-05-12 contra Nano del proyecto:**

```bash
$ cat /etc/nv_tegra_release
# R32 (release), REVISION: 7.1, GCID: 29818004, BOARD: t210ref,
# EABI: aarch64, DATE: Sat Feb 19 17:05:08 UTC 2022
$ uname -r
4.9.253-tegra
```

→ **JP 4.6.1 (R32.7.1) con build de feb 2022 tiene kernel 4.9.253-tegra**, idéntico al kernel de R32.6.1. NVIDIA no bumpeó el patch level entre R32.6.1 y la primera versión de R32.7.1.

| JetPack | L4T | Build | Kernel `uname -r` |
|---|---|---|---|
| 4.6 | R32.6.1 | ago 2021 | `4.9.253-tegra` |
| **4.6.1** | **R32.7.1** | **feb 2022** | **`4.9.253-tegra`** ← Nano del proyecto |
| 4.6.1 | R32.7.1 | builds posteriores | `4.9.253-tegra` o `4.9.299-tegra` según fecha de flash |
| 4.6.2 | R32.7.2 | abr 2022 | `4.9.253-tegra` o `4.9.299-tegra` |
| 4.6.3 | R32.7.3 | nov 2022 | `4.9.337-tegra` |
| 4.6.4 | R32.7.4 | jun 2023 | `4.9.337-tegra` |

**Implicación corregida:** el handoff `HANDOFF-implementacion-vastai-hf.md` afirmaba "JetPack 4.6.1 (kernel 4.9.337)". La parte **JetPack 4.6.1 es CORRECTA**. La parte **kernel 4.9.337 es INCORRECTA** — el kernel real de la Nano del proyecto es `4.9.253-tegra` (R32.7.1 build feb 2022). El `4.9.337-tegra` sólo aparece en builds de JP 4.6.3+ que el usuario no tiene.

**Comando de verificación (canon):**

```bash
cat /etc/nv_tegra_release
# JP 4.6:   # R32 (release), REVISION: 6.1, ...
# JP 4.6.1: # R32 (release), REVISION: 7.1, ...
# JP 4.6.3: # R32 (release), REVISION: 7.3, ...
```

**Notas low-confidence:** el patch level exacto que se bumpea entre builds intermedios de R32.7.1 / R32.7.2 no está documentado verbatim por NVIDIA en docs públicas. La afirmación "4.9.299 en builds posteriores" es inferencia de foros con baja confianza; lo confirmado con primera mano es **4.9.253 en R32.7.1 build feb 2022**.

### 21.3 Veredicto: headers en apt

**`linux-headers-$(uname -r)` NO existe en el repo APT de NVIDIA para Jetson en JetPack 4.6.x.** Confirmado:

- Foro NVIDIA mayo 2022 [#213532](https://forums.developer.nvidia.com/t/unable-to-locate-package-linux-headers-4-9-253-tegra/213532): `E: Unable to locate package linux-headers-4.9.253-tegra`.
- Foro NVIDIA septiembre 2023 [#265546](https://forums.developer.nvidia.com/t/install-linux-headers-on-jetson-nano-failed/265546), moderador NVIDIA `DaveYYY` verbatim: *"kernel header files should already be included in `/usr/src/`, and is there any reason you have to install it separately?"*

El paquete que SÍ existe es `nvidia-l4t-kernel-headers`. Al instalarlo, deja headers en `/usr/src/linux-headers-<KVER>-tegra-ubuntu18.04_aarch64/kernel-4.9/`. Pero ese path **no coincide con la convención que DKMS busca** (`/lib/modules/$(uname -r)/build`), de ahí el error original. Y `getKernelSources.sh` de JetsonHacks falla cuando intenta copiar `Module.symvers` desde ese path si el paquete no está instalado.

### 21.4 Análisis del script `getKernelSources.sh`

Verificado vía `gh api` sobre el repo `JetsonHacksNano/buildKernelAndModules` (sufijo: `scripts/getKernelSources.sh`):

| Paso | Acción |
|---|---|
| 1 | Lee `JETSON_L4T` y `JETSON_CHIP_ID` de `scripts/jetson_variables` |
| 2 | Selecciona URL del tarball según L4T y board (`t210ref` para Nano) |
| 3 | Para L4T 32.6.1: `https://developer.nvidia.com/embedded/l4t/r32_release_v6.1/sources/t210/public_sources.tbz2` |
| 4 | Descarga en `/usr/src/` (configurable con `-d`) |
| 5 | Extrae sólo `kernel_src.tbz2` del tarball outer, luego descomprime |
| 6 | Elimina tarballs intermedios |
| 7 | **Copia `Module.symvers` desde `/usr/src/linux-headers-<KVER>-tegra-ubuntu18.04_aarch64/kernel-4.9/`** ← este paso FALLA si el paquete `nvidia-l4t-kernel-headers` no está instalado |
| 8 | Clona `.config` activo: `zcat /proc/config.gz > .config` |
| 9 | Fija `LOCALVERSION="-tegra"` en `.config` |
| 10 | **NO abre menuconfig** automáticamente |

**Path final del source:** `/usr/src/kernel/kernel-4.9/` (constante, validado en `scripts/getKernelSources.sh` línea con `PROPOSED_SRC_PATH="$SOURCE_TARGET/kernel/kernel-$KERNEL_RELEASE"`).

El fallo `cp: cannot stat '...Module.symvers'` del error que vimos confirma que el paso 7 es el que rompe. **La pérdida de `Module.symvers` no es fatal** porque podemos regenerarlo con `make modules_prepare`.

### 21.5 Sintaxis verificada de `dkms --kernelsourcedir`

`man dkms` 2.3 (Ubuntu Bionic, la versión que trae JetPack 4.6.x):

```
--kernelsourcedir <kernel-source-directory-location>
    Using this option you can specify the location of your kernel
    source directory. Most likely you will not need to set this if
    your kernel source is accessible via /lib/modules/$kernel_version/build.
```

- Sintaxis: `--kernelsourcedir /path/al/dir` (espacio entre flag y valor).
- Path al directorio que contiene el `Makefile` del kernel (raíz del source tree).
- Se pasa en `build` y/o `install` (idéntico en DKMS 2.x y 3.x).
- **`install` llama internamente a `build`, así que basta con pasarlo una vez en `install`.**

### 21.6 Receta consolidada (R7) — desbloqueo en 6 pasos

```bash
# 0. Confirmar versión JetPack (debería decir R32, REVISION: 6.1 para JP 4.6)
cat /etc/nv_tegra_release

# 1. Verificar que el source descargado por getKernelSources.sh está donde toca
ls /usr/src/kernel/kernel-4.9/Makefile
# Si no existe, primero correr:
#   cd ~/buildKernelAndModules && sudo ./getKernelSources.sh
# (ignorar el "cp: cannot stat ... Module.symvers" — lo regeneramos en paso 3)

# 2. Dependencias para preparar el kernel
sudo apt install -y libssl-dev libelf-dev bison flex bc

# 3. Preparar el kernel source: genera Module.symvers desde la config viva
cd /usr/src/kernel/kernel-4.9
sudo zcat /proc/config.gz | sudo tee .config >/dev/null
sudo make ARCH=arm64 olddefconfig
sudo make ARCH=arm64 modules_prepare
# Toma 3-5 min en Nano. Output debe terminar sin "bad exit status"

# 4. Crear el symlink canónico que DKMS busca por defecto
sudo rm -f /lib/modules/$(uname -r)/build
sudo ln -sf /usr/src/kernel/kernel-4.9 /lib/modules/$(uname -r)/build

# 5. Verificación
ls -la /lib/modules/$(uname -r)/build       # symlink → /usr/src/kernel/kernel-4.9
ls /lib/modules/$(uname -r)/build/Makefile  # debe existir

# 6. Limpiar el intento previo fallido y reinstalar (sin --kernelsourcedir, lo resuelve el symlink)
sudo dkms remove 8188eu/5.3.9 --all 2>/dev/null || true
sudo dkms install 8188eu/5.3.9

# 7. Verificar éxito
dkms status                                 # debe mostrar: 8188eu, 5.3.9: installed
# Tras conectar la TL-WN722N v4:
sudo modprobe 8188eu
lsmod | grep 8188eu                          # debe mostrar: 8188eu (sin la r delante)
iw list | grep -A8 "Supported interface modes"  # debe listar * AP
```

### 21.7 Fallback explícito si el symlink no funciona

Si en `/lib/modules/$(uname -r)/` no hay directorio (Nano no lo crea por default en algunos casos), crear primero:

```bash
sudo mkdir -p /lib/modules/$(uname -r)
sudo ln -sf /usr/src/kernel/kernel-4.9 /lib/modules/$(uname -r)/build
```

Y como último recurso, usar `--kernelsourcedir` explícito:

```bash
KVER=$(uname -r)
sudo dkms add -m 8188eu -v 5.3.9 2>/dev/null || true
sudo dkms build   -m 8188eu -v 5.3.9 -k $KVER --kernelsourcedir /usr/src/kernel/kernel-4.9
sudo dkms install -m 8188eu -v 5.3.9 -k $KVER --kernelsourcedir /usr/src/kernel/kernel-4.9
```

### 21.8 Actualización a Decisión D16

> **D16 — versión R7:** cliente Wi-Fi USB en Nano: TP-Link TL-WN722N v4 (RTL8188EUS) + driver `aircrack-ng/rtl8188eus@v5.3.9` instalado vía DKMS. **Requisito previo al DKMS:** bajar kernel source con `getKernelSources.sh` (JetsonHacksNano/buildKernelAndModules) → preparar con `zcat /proc/config.gz > .config && make modules_prepare ARCH=arm64` → crear symlink `/lib/modules/$(uname -r)/build → /usr/src/kernel/kernel-4.9`. Sin estos 3 pasos previos, DKMS falla con "kernel headers cannot be found" porque NVIDIA NO distribuye `linux-headers-X-tegra` vía apt. Aplicar `blacklist r8188eu` antes del reboot.

### 21.9 Fuentes Ronda 7

| # | Título | URL | Tipo |
|---|---|---|---|
| 45 | NVIDIA Jetson Linux R32.7.1 release page | https://developer.nvidia.com/embedded/linux-tegra-r3271 | Doc oficial |
| 46 | NVIDIA Jetson Linux R32.6.1 release page | https://developer.nvidia.com/embedded/linux-tegra-r3261 | Doc oficial |
| 47 | NVIDIA Jetson Linux R32.7.3 release page | https://developer.nvidia.com/embedded/linux-tegra-r3273 | Doc oficial |
| 48 | NVIDIA Jetson Linux R32.7.4 release page | https://developer.nvidia.com/embedded/linux-tegra-r3274 | Doc oficial |
| 49 | NVIDIA Forums #213532 — `linux-headers-4.9.253-tegra` apt fail | https://forums.developer.nvidia.com/t/unable-to-locate-package-linux-headers-4-9-253-tegra/213532 | Foro |
| 50 | NVIDIA Forums #265546 — install linux-headers fail (DaveYYY) | https://forums.developer.nvidia.com/t/install-linux-headers-on-jetson-nano-failed/265546 | Foro |
| 51 | NVIDIA Forums #197863 — kernel version in JetPack4.6 | https://forums.developer.nvidia.com/t/what-is-the-kernel-version-in-jetpack4-6/197863 | Foro |
| 52 | Ubuntu Manpages — `dkms(8)` Bionic v2.3 | https://manpages.ubuntu.com/manpages/bionic/man8/dkms.8.html | Doc oficial |
| 53 | RidgeRun wiki — Building Jetson Nano Kernel | https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Nano_-_Building_the_Kernel_from_Source | Wiki técnica |
| 54 | `JetsonHacksNano/buildKernelAndModules` (verificado vía gh) | https://github.com/JetsonHacksNano/buildKernelAndModules | Repo |
| 55 | `jetsonhacks/jetson-linux-build/scripts/getKernelSources.sh` (sucesor) | https://github.com/jetsonhacks/jetson-linux-build/blob/main/scripts/getKernelSources.sh | Repo |

### 21.10 Gaps de evidencia residuales R7

- Versión exacta `4.9.299` para R32.7.1 / R32.7.2 inferida de pattern de foros, **no confirmada con doc oficial NVIDIA** que sólo dice "Linux Kernel 4.9" sin el número de parche. Confianza media.
- El usuario debería ejecutar `cat /etc/nv_tegra_release` para confirmar si efectivamente tiene JP 4.6 (R32.6.1) o si el handoff tenía razón en la versión JP pero no en el kernel. En cualquier caso, la receta §21.6 funciona para cualquier kernel JP 4.6.x.

### 21.11 Historial actualizado

| Ronda | Fecha | Profundidad | Foco |
|---|---|---|---|
| 6 | 2026-05-12 | Alto | Acceso remoto Win11 → Nano + TL-WN722N v4 + Contabo bastion + overlay networking |
| 7 | 2026-05-12 | Bajo | Fix DKMS install RTL8188EUS — corrección kernel version + path getKernelSources + sintaxis `--kernelsourcedir` |
| **7-bis** | **2026-05-12** | **In-situ (sin nueva ronda)** | **Cierre empírico — driver real, USB ID real, WiFi Intel descubierto** (§22) |

---

## 22. Cierre empírico R7 — hallazgos de la implementación in-situ (2026-05-12)

> Esta sección documenta lo que **realmente pasó** al ejecutar la receta §21.6 contra la Nano del proyecto, después de un `sudo apt upgrade` que bumpeó implícitamente el kernel, conectar físicamente la antena USB, y resolver dos errores de compilación que la R7 teórica no podía anticipar. Genera 4 decisiones nuevas vinculantes (D21-D24) que reemplazan asunciones de R6 + R7.

### 22.1 Kernel real tras `apt upgrade` — flip implícito a JP 4.6.5

Al ejecutar `sudo apt upgrade` antes del DKMS, el paquete `nvidia-l4t-kernel` se actualizó de `4.9.253-tegra-32.7.1` (build feb 2022, JP 4.6.1) a `4.9.337-tegra-32.7.6` (build 2024-11-04, JP 4.6.5) **sin que el usuario hubiera bumpeado JetPack explícitamente**. El repo APT de NVIDIA sirve el patch level más alto disponible dentro de R32.x sin requerir cambio mayor de versión.

```bash
$ uname -r                          # antes del reboot:
4.9.253-tegra
$ dpkg -l | grep nvidia-l4t-kernel  # paquete instalado:
ii  nvidia-l4t-kernel    4.9.337-tegra-32.7.6-20241104234540
# Tras reboot:
$ uname -r
4.9.337-tegra
```

**Consecuencia operacional:** el path `/lib/modules/4.9.337-tegra/build` aparece automáticamente como symlink válido a `/usr/src/linux-headers-4.9.337-tegra-ubuntu18.04_aarch64/kernel-4.9/`. **La receta §21.6 (getKernelSources + make modules_prepare + symlink manual) se vuelve INNECESARIA** cuando se hace upgrade primero.

### 22.2 Receta condensada empírica (sustituye §21.6)

```bash
# 0. ANTES de DKMS: asegurar último patch del kernel y reboot
sudo apt update && sudo apt upgrade -y
sudo reboot
# Verificar tras reboot:
uname -r                                          # debe coincidir con dpkg -l nvidia-l4t-kernel
ls -la /lib/modules/$(uname -r)/build             # debe ser symlink válido (no rojo)
ls /lib/modules/$(uname -r)/build/Makefile        # debe existir

# 1. Compilar el driver — NO usar aircrack-ng/rtl8188eus, ver §22.4
git clone https://github.com/lwfinger/rtl8188eu.git
cd rtl8188eu
make ARCH=arm64
sudo make install
sudo depmod -a

# 2. Blacklist staging in-kernel
sudo tee /etc/modprobe.d/blacklist-realtek.conf >/dev/null <<'EOF'
blacklist r8188eu
blacklist rtl8xxxu
EOF
sudo update-initramfs -u

# 3. Conectar la antena USB físicamente (puerto USB 2.0 negro, no 3.0 azul)
sudo modprobe 8188eu

# 4. Verificar
lsmod | grep 8188eu                       # 8188eu (sin la "r" delante) con uso >= 1
ls /sys/class/ieee80211/                  # debe listar phy0/phy1
iw dev                                    # debe listar wlanX con type managed
dmesg | grep -i "CHIP_8188E"              # debe ver "CHIP_8188E_Normal_Chip_TSMC_D_CUT"
```

### 22.3 USB ID real del TL-WN722N v4 — corrección a §3

El doc R6 §3.1 y §3.2 asumía USB ID `0bda:8179` (Realtek vendor genérico). **Verificado empíricamente:**

```bash
$ lsusb | grep -i "802.11\|realtek\|TP-Link"
Bus 001 Device 014: ID 2357:010c
$ lsusb -v -d 2357:010c 2>/dev/null | grep -E "iProduct|iManufacturer"
  iManufacturer  1 Realtek
  iProduct       2 802.11n NIC
$ dmesg | grep "Chip Version"
[ 1690.260723] Chip Version Info: CHIP_8188E_Normal_Chip_TSMC_D_CUT_1T1R_RomVer(0)
```

- **VID:PID:** `2357:010c` (TP-Link vendor) — no `0bda:*` (Realtek vendor)
- **Chip silicon revision:** RTL8188E rev D (TSMC), single-stream 1T1R
- **Producto USB string:** "802.11n NIC", manufacturer "Realtek"

El v3 tiene PID `2357:010d`, y algunos lotes raros del v4 muestran PID `2357:0111`. Los IDs `0bda:8179`/`0bda:0179` aparecen sólo cuando el adapter es de marca blanca sin sticker TP-Link y reporta directamente Realtek vendor — no es el caso de los TL-WN722N v4 retail.

### 22.4 El driver `aircrack-ng/rtl8188eus@v5.3.9` NO compila en kernel 4.9-tegra

Tras seguir §21.6 al pie de la letra (kernel preparado, symlinks correctos, headers válidos), el `dkms install 8188eu/5.3.9` se ejecutó hasta `make[1]: Entering directory '/usr/src/linux-headers-4.9.337-tegra-ubuntu18.04_aarch64/kernel-4.9'` y compiló 40+ archivos correctamente, pero **falló específicamente en `os_dep/linux/ioctl_cfg80211.c:1150-1155`**:

```c
// Línea 1155 en ioctl_cfg80211.c — driver source assume kernel >= 4.14:
cfg80211_connect_bss(wdev_to_ndev(pwdev), cur_network->network.MacAddress, bss,
                     ..., GFP_ATOMIC, NL80211_TIMEOUT_UNSPECIFIED);
//                          ↑ esto NO existe en kernel 4.9
```

**Error literal del log:**

```
ioctl_cfg80211.c:1155:68: error: 'NL80211_TIMEOUT_UNSPECIFIED' undeclared (first use in this function);
                                  did you mean 'NL80211_IFTYPE_UNSPECIFIED'?
ioctl_cfg80211.c:1150:25: error: too many arguments to function 'cfg80211_connect_bss'
./include/net/cfg80211.h:4986:6: note: declared here
 void cfg80211_connect_bss(struct net_device *dev, const u8 *bssid, ...
```

**Causa raíz:** la enum `nl80211_timeout_reason` con valor `NL80211_TIMEOUT_UNSPECIFIED` se introdujo en upstream kernel **4.14** (`include/uapi/linux/nl80211.h`, commit `bf1ecd2`), junto con un parámetro extra para `cfg80211_connect_bss()`. NVIDIA nunca rebaseó tegra-4.9 más allá del baseline 4.9 + parches selectos. El fork `aircrack-ng/rtl8188eus@v5.3.9` asume kernels ≥4.14 sin guardar la llamada con `#if LINUX_VERSION_CODE >= KERNEL_VERSION(4,14,0)`.

**Alcance del bug:** afecta a `aircrack-ng/rtl8188eus` y a su fork `kimocoder/rtl8188eus`. Parches manuales son posibles (rodear con `#if/#else/#endif`) pero el archivo tiene varias llamadas similares y el fix in-place suele cascadear.

### 22.5 Driver canónico que SÍ funciona: `lwfinger/rtl8188eu`

**Verificado empíricamente:**

```bash
$ git clone https://github.com/lwfinger/rtl8188eu.git
$ cd rtl8188eu && make ARCH=arm64
# ... compila clean en ~3-4 min sobre Nano (4 cores @ 1.43 GHz) ...
$ sudo make install && sudo depmod -a
$ sudo modprobe 8188eu && lsmod | grep 8188eu
8188eu                945147  0
$ dmesg | tail
[ 1021.107270] R8188EU: Firmware Version 11, SubVersion 1, Signature 0x88e1
```

**Por qué funciona:** Larry Finger es kernel-dev mainline de Realtek; sus drivers out-of-tree mantienen wrappers `LINUX_VERSION_CODE` para kernels 3.x-6.x sin discriminar. No requiere patches.

**Limitación vs aircrack-ng:** `lwfinger/rtl8188eu` no incluye packet injection avanzado (airodump-ng -i forzado, aireplay-ng deauth dirigido). **No es nuestro caso de uso** — necesitamos STA cliente y AP mode con hostapd, ambos soportados nativamente por lwfinger.

**Mito desmentido — `morrownr` NO mantiene fork para 8188eu:** El user `morrownr` (4 K stars en USB-WiFi guide) cubre 8821cu, 8821au, 8814au, 8812au, 88x2bu, 88x2cu — pero **NO 8188eu/8188eus**. La línea 8188 sólo tiene drivers out-of-tree de `lwfinger`, `aircrack-ng`, `kimocoder`, y `anusornint` (fork lwfinger). Recomendación de morrownr en su USB-WiFi guide: si necesitás 8188eu, usar el staging in-kernel o lwfinger.

### 22.6 Hallazgo lateral — Wi-Fi Intel interno ya presente en la Nano

Al ejecutar `ip link show` y `nmcli device status` durante el debug, apareció un wlan0 que **no era la antena USB**:

```bash
$ ip link show wlan0
10: wlan0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 ... state DOWN ...
    link/ether 3c:64:cf:d9:6d:51 brd ff:ff:ff:ff:ff:ff
$ nmcli device status
DEVICE   TYPE      STATE         CONNECTION        
eth0     ethernet  connected     Wired connection 1
wlan0    wifi      disconnected  --                
```

- **MAC OUI:** `3c:64:cf` = **Intel Corporate** (verificable en [OUI lookup IEEE](https://standards-oui.ieee.org/oui/oui.txt))
- **Slot:** M.2 Key E del Nano B01 (la Nano base no trae chip Wi-Fi; alguien instaló módulo M.2 a la placa, probablemente Intel AC8265 o similar — confirmable con `lshw -C network -short`)
- **Driver:** in-tree `iwlwifi` (parte del kernel mainline, NVIDIA lo trae habilitado en tegra-4.9)

**Consecuencia estratégica:** la Nano tiene **dos** chips Wi-Fi disponibles, no uno. Esto invalida la asunción de R6 §3-§5 de que el RTL8188EUS USB era el único Wi-Fi.

### 22.7 Estrategia dual de Wi-Fi (reemplaza D16 + D19)

> **D21 — Driver Wi-Fi USB definitivo (reemplaza D16):**
> Para el TP-Link TL-WN722N v4 (RTL8188EUS) en Jetson Nano kernel 4.9-tegra (cualquier patch level 4.9.253 a 4.9.337+), el driver out-of-tree canónico es **`lwfinger/rtl8188eu`** (sin la "s"), no `aircrack-ng/rtl8188eus`. El segundo no compila por API mismatch documentado en §22.4. Instalación vía `make && make install` (no DKMS por default; agregar a DKMS opcional). Blacklist obligatorio de `r8188eu` (staging) y `rtl8xxxu` (mainline alternativa).

> **D22 — Estrategia dual de Wi-Fi (extiende D17, D19):**
> Aprovechar los dos chips Wi-Fi de la Nano del proyecto:
> - **Intel M.2 (wlan0, driver iwlwifi in-tree)** → **cliente STA** para conectar la Nano al Wi-Fi del salón/UAO/hotspot móvil durante dev y demo. NetworkManager lo maneja nativamente; Tailscale corre encima.
> - **TP-Link TL-WN722N v4 USB (wlanX nuevo, driver lwfinger/rtl8188eu)** → **AP mode hostapd** standalone, sólo en el escenario battery-powered sin Internet del deploy final. Activable bajo demanda con un perfil NetworkManager guardado (ver §14 escenario C, ajustando ifname al de lwfinger en runtime).
>
> Esto elimina el riesgo de quedarse sin Wi-Fi si el USB falla o el chip Intel no estuviera, y permite usar la antena USB exclusivamente para el rol especializado (AP) sin competir con STA.

> **D23 — Pre-requisito de cualquier DKMS para driver out-of-tree (reemplaza §21.6 paso 1-5):**
> Antes de instalar cualquier driver Realtek/Realtek-like vía DKMS o `make install` en Jetson Nano JP 4.6.x, ejecutar `sudo apt update && sudo apt upgrade -y && sudo reboot` para bumpear `nvidia-l4t-kernel` al patch level más reciente del repo NVIDIA (`4.9.337-tegra-32.7.6` a mayo 2026). Esto provee `nvidia-l4t-kernel-headers` con symlink válido en `/lib/modules/$(uname -r)/build`, eliminando la necesidad de `getKernelSources.sh` + `make modules_prepare` manual (§21.6 pasos 1-5). El `apt upgrade` es seguro a este nivel porque NVIDIA garantiza compat dentro de R32.x.

> **D24 — Identificadores hardware verificados (corrige §3.1, §3.2):**
> - **TL-WN722N v4 USB VID:PID = `2357:010c`** (TP-Link vendor), no `0bda:8179`.
> - **Chip silicon = RTL8188E rev D TSMC 1T1R** (`CHIP_8188E_Normal_Chip_TSMC_D_CUT`).
> - **Wi-Fi interno = Intel M.2 Key E**, MAC OUI `3c:64:cf` (Intel Corporate).

### 22.8 Patrón dmesg "normal" tras `modprobe 8188eu` (referencia)

Para distinguir "el driver funciona y está esperando AP/conexión" vs "el driver falló":

```
[XXXX] usb 1-2.X: new high-speed USB device number Y using tegra-xusb
[XXXX] usb 1-2.X: New USB device found, idVendor=2357, idProduct=010c
[XXXX] usb 1-2.X: Product: 802.11n NIC
[XXXX] usb 1-2.X: Manufacturer: Realtek
[XXXX] Chip Version Info: CHIP_8188E_Normal_Chip_TSMC_D_CUT_1T1R_RomVer(0)
[XXXX] R8188EU: Firmware Version 11, SubVersion 1, Signature 0x88e1
[XXXX] IPv6: ADDRCONF(NETDEV_UP): wlanX: link is not ready    ← normal, no es error
[XXXX] R8188EU: INFO indicate disassoc                         ← normal, sólo dice "no asociada a AP"
```

**El mensaje `R8188EU: INFO indicate disassoc` es informativo**, no error. Significa "no estoy asociada a ningún AP", lo que es correcto inmediatamente después de cargar el módulo. NO indicar fallo de inicialización ni desconexión USB.

**Distinguir de fallo real**: si después de `modprobe` no aparece ningún `Firmware Version` en dmesg, o si `ls /sys/class/ieee80211/` queda vacío después de >5s, ahí sí hay falla. Si los `Firmware Version` + `wlanX` aparecen, el driver está OK aunque `iw dev` esté momentáneamente vacío durante el bring-up de NetworkManager.

### 22.9 Fuentes Ronda 7-bis (cierre empírico)

| # | Título | URL | Tipo |
|---|---|---|---|
| 56 | lwfinger/rtl8188eu (driver canónico verificado in-situ) | https://github.com/lwfinger/rtl8188eu | Repo |
| 57 | aircrack-ng/rtl8188eus default branch v5.3.9 (driver NO compila k4.9) | https://github.com/aircrack-ng/rtl8188eus | Repo |
| 58 | Kernel commit bf1ecd2 — NL80211_TIMEOUT_UNSPECIFIED introducido en 4.14 | https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=bf1ecd2 | Commit upstream |
| 59 | morrownr/USB-WiFi (guía de referencia comunitaria) | https://github.com/morrownr/USB-WiFi | Repo |
| 60 | OUI lookup IEEE (Intel `3c:64:cf` verificación) | https://standards-oui.ieee.org/oui/oui.txt | Doc oficial |
| 61 | NVIDIA L4T R32.7.6 release notes (kernel 4.9.337-tegra-32.7.6) | https://developer.nvidia.com/embedded/jetson-linux-r3276 | Doc oficial |

### 22.10 Tareas pendientes derivadas de §22

1. **Validar conexión STA con wlan0 Intel**: `nmcli device wifi list`, conectar al SSID del salón, medir RTT/throughput. Determina si el chip M.2 es estable.
2. **Verificar AP mode con lwfinger driver** (no aircrack-ng): `iw list` debe listar "AP" en supported interface modes para el wlanX del USB.
3. **Reajustar §14 install runbook**: cambiar repo del driver de `aircrack-ng/rtl8188eus` a `lwfinger/rtl8188eu`, eliminar pasos de getKernelSources.sh, agregar `apt upgrade + reboot` como paso 0.
4. **Actualizar handoff `HANDOFF-implementacion-vastai-hf.md`**: el kernel real ahora es 4.9.337-tegra-32.7.6 (JP 4.6.5 implícito) tras el apt upgrade, no 4.9.253 ni la confusión previa.
5. **Próxima Ronda 8**: foco en implementación final del stack (Tailscale + NoMachine + Xfce4 + dummy HDMI + hostapd con lwfinger driver).

---

## 23. Ronda 8 — implementación final del stack acceso remoto (2026-05-12, profundidad media)

### 23.1 Motivación y resultados clave

R8 cierra los 4 huecos de implementación que quedaban: (1) hostapd config exacta para `lwfinger/rtl8188eu`, (2) NoMachine + Xfce4 + dummy HDMI en JP 4.6.5, (3) Tailscale en Bionic ARM64 con bug DNS conocido, (4) autossh vs alternativas modernas. **Aporta 4 decisiones nuevas (D25-D28)** que reemplazan/refinan asunciones de R6.

**Hallazgos vinculantes resumidos:**

| Tema | Hallazgo decisivo | Decisión |
|---|---|---|
| hostapd + lwfinger | Issue #363 confirma AP mode funcional en Jetson Nano con rama `v5.2.2.4`, config `driver=rtl871xdrv` (NO `nl80211`), `hw_mode=g`, ieee80211n opcional | D25 |
| NoMachine + Xfce4 | KB AR02R01074 actualizado 2025-07; receta probada en JP 4.6.5; dummy HDMI plug físico = $5-8, cero config; `DefaultDesktopCommand "/usr/bin/startxfce4"` | D26 |
| Tailscale Bionic | Bug ARM64 #14902 (`dns: [rc=unknown ret=direct]`) activo desde 1.78 hasta 1.84+; workaround: `--accept-dns=false` + symlink resolv.conf → systemd-resolved stub-resolv | D27 |
| autossh vs alternativas | autossh sigue siendo la opción de menor riesgo dado que el VPS Contabo ya existe; chisel v1.11.5 ARM64 .deb es backup viable; Cloudflared = único bypass real si UAO bloquea SSH | D28 |

### 23.2 hostapd + dnsmasq receta verificada para `lwfinger/rtl8188eu`

**Fuente primaria:** [Issue #363 lwfinger/rtl8188eu](https://github.com/lwfinger/rtl8188eu/issues/363) (alpop, 2021-03-02, cerrado por lwfinger). Reporte de primera mano confirmando AP mode estable con TL-WN722N V3 (mismo chipset que nuestro v4) en Jetson Nano 2GB JetPack 4.5. lwfinger respondió "Makefile fixed today" — el master actual ya incluye el fix aarch64.

**Cambio de rama relevante:** la rama master tiene compat ARM64 pero algunos reportes de inestabilidad en hostapd. La rama `v5.2.2.4` es la combinación más conservadora reportada funcionando. **Recomendación:** intentar primero con master (que es lo que ya tenés instalado), si hostapd da problemas con `driver=rtl871xdrv`, switch a `v5.2.2.4`.

**Config `hostapd.conf` canónica del repo** (verificada en `lwfinger/rtl8188eu/rtl_hostapd.conf` master):

```ini
# /etc/hostapd/hostapd.conf — para uso con driver out-of-tree lwfinger/rtl8188eu
interface=wlan1                  # ajustar al nombre real del USB (ip link show)
ctrl_interface=/var/run/hostapd
ssid=embebidos3-nano
channel=6                        # canal 2.4 GHz fijo (NO auto)
wpa=2
wpa_passphrase=<PSK_FUERTE>
driver=rtl871xdrv                # CRÍTICO: NO usar nl80211 con este driver
beacon_int=100
hw_mode=g                        # NO 'a' (no soporta 5 GHz), NO 'n' inicial
# ieee80211n=1                   # comentado — si hay desconexiones tras 10-15min, dejar comentado
# wme_enabled=1                  # idem
# ht_capab=[SHORT-GI-20]         # solo activar si ieee80211n=1
wpa_key_mgmt=WPA-PSK
wpa_pairwise=CCMP
rsn_pairwise=CCMP
max_num_sta=8
wpa_group_rekey=86400
```

**Crítico — `driver=rtl871xdrv` no `nl80211`:** issue #344 y #456 documentan que con `driver=nl80211` el chip RTL8188EUS devuelve `nl80211: Driver does not support authentication/association` y los canales aparecen con `max_tx_power=0 dBm`. El driver propietario `rtl871xdrv` (que viene en el repo `lwfinger/rtl8188eu`) es el único que funciona vía hostapd para AP mode.

**hostapd que sí compila contra `rtl871xdrv`:** el `apt install hostapd` de Bionic NO incluye soporte para `rtl871xdrv` driver. Hay que usar el hostapd que viene en `lwfinger/rtl8188eu/hostapd-2.9/` y compilarlo:

```bash
cd ~/rtl8188eu/hostapd-2.9/hostapd
cp defconfig .config
# Editar .config y añadir:
echo "CONFIG_DRIVER_RTW=y" >> .config
make -j4
sudo cp hostapd /usr/local/sbin/hostapd-rtw
# Usar /usr/local/sbin/hostapd-rtw en vez de /usr/sbin/hostapd
```

**dnsmasq config para servir DHCP a clientes del AP:**

```ini
# /etc/dnsmasq.d/embebidos3-ap.conf
interface=wlan1
bind-interfaces
dhcp-range=192.168.42.10,192.168.42.50,12h
dhcp-option=3,192.168.42.1       # gateway
dhcp-option=6,192.168.42.1       # DNS (la Nano)
domain-needed
bogus-priv
no-resolv
server=1.1.1.1
server=8.8.8.8
```

**IP estática en wlan1 antes de hostapd start:**

```bash
sudo ip addr add 192.168.42.1/24 dev wlan1
sudo ip link set wlan1 up
sudo /usr/local/sbin/hostapd-rtw -B /etc/hostapd/hostapd.conf
sudo systemctl restart dnsmasq
```

**Throughput esperado:** RTL8188E rev D 1T1R en hw_mode=g (sin n), 2.4 GHz, single stream → **techo teórico 54 Mbps, real 15-20 Mbps UDP** con cliente único. Suficiente para inferencia remota.

### 23.3 NoMachine + Xfce4 + dummy HDMI en JetPack 4.6.5

**Fuente primaria:** KB NoMachine AR02R01074 (actualizado 2025-07-03, válido para JP 4.6.5). Walkthrough complementario: [JetsonHacks blog 2023-12](https://jetsonhacks.com/2023/12/03/nomachine-jetson-remote-desktop/) + Connect Tech soporte técnico.

**Receta consolidada (R8 §23.3 vs R6 §13):**

```bash
# === 1. Xfce4 + dependencias (NO descomenta xdg-utils en bionic, ya viene) ===
sudo apt update
sudo apt install -y xfce4 xfce4-goodies xfce4-terminal

# === 2. Login automático en Xfce4 para autostart ===
sudo mkdir -p /etc/lightdm
sudo tee /etc/lightdm/lightdm.conf.d/12-autologin.conf <<EOF
[Seat:*]
autologin-user=$USER
autologin-user-timeout=0
user-session=xfce
EOF

# === 3. NoMachine ARM64 (verificar última versión en https://downloads.nomachine.com/linux/) ===
cd /tmp
# Reemplazar la URL del .deb con la más reciente arm64:
wget "https://download.nomachine.com/download/8.16/Arm/nomachine_8.16.1_2_arm64.deb" -O nomachine.deb
sudo dpkg -i nomachine.deb || sudo apt-get install -f -y

# === 4. Forzar Xfce4 como default desktop en NoMachine ===
sudo sed -i 's|^#DefaultDesktopCommand .*|DefaultDesktopCommand "/usr/bin/startxfce4"|' /usr/NX/etc/node.cfg
# Si la línea no existe (algunas versiones), agregarla:
grep -q "^DefaultDesktopCommand" /usr/NX/etc/node.cfg || \
  echo 'DefaultDesktopCommand "/usr/bin/startxfce4"' | sudo tee -a /usr/NX/etc/node.cfg

# === 5. Restart NoMachine ===
sudo /usr/NX/bin/nxserver --restart

# === 6. Cambiar el target default a multi-user (NO graphical), opcional ===
# Sólo si querés que la Nano NO arranque sesión X cuando no hay HDMI conectado
# y solo deje NoMachine activo. Recomendado con dummy plug puesto:
# sudo systemctl set-default multi-user.target
# Reboot para aplicar
```

**Dummy HDMI plug:** la receta más simple es **comprar un dummy HDMI físico** ($5-8 en Amazon, "headless ghost adapter", resolución 1920x1080). Plug-and-play, cero config. Funciona en Jetson Nano sin modificación de Xorg.

**Alternativa software (sin dummy físico):** virtual display con `xserver-xorg-video-dummy`. Funciona pero en Jetson Nano el driver `tegra` no siempre cede control al driver `dummy` limpiamente. Walkthrough en Amplifi Labs blog ([amplifilabs.com/post/nomachine-with-xfce-desktop-on-headless-vps-complete-setup-guide](https://www.amplifilabs.com/post/nomachine-with-xfce-desktop-on-headless-vps-complete-setup-guide), 2026-03-30):

```bash
sudo apt install xserver-xorg-video-dummy
sudo tee /etc/X11/xorg.conf.d/10-dummy.conf <<'EOF'
Section "Device"
    Identifier "Dummy"
    Driver "dummy"
    VideoRam 256000
EndSection
Section "Monitor"
    Identifier "Monitor0"
    HorizSync 28.0-80.0
    VertRefresh 48.0-75.0
    Modeline "1920x1080" 148.5 1920 2008 2052 2200 1080 1084 1089 1125 +hsync +vsync
EndSection
Section "Screen"
    Identifier "Screen0"
    Device "Dummy"
    Monitor "Monitor0"
    DefaultDepth 24
    SubSection "Display"
        Depth 24
        Modes "1920x1080"
    EndSubSection
EndSection
EOF
```

**Reporte de regresión post-JP 4.6.5:** ninguno reportado a mayo 2026. La KB de NoMachine sigue mostrando "Last Update: 2025-07-03" sin actualización para 4.6.5 específica; la receta es estable.

**Video de referencia 2024 (verificable):** [make2explore Systems — Tutorial Installation of NoMachine on NVIDIA Jetson Nano](https://youtu.be/vBMHS6FXBM4), 2024-10-18, 18:41 min, chapters disponibles:

- `0:00-0:31` Start
- `0:31-0:43` Introduction
- `0:43-3:15` What is NoMachine RDS Tool?
- `3:15-5:25` Main Objective ([https://youtu.be/vBMHS6FXBM4?t=195](https://youtu.be/vBMHS6FXBM4?t=195))
- `5:25-6:43` Installing JTOP ([https://youtu.be/vBMHS6FXBM4?t=325](https://youtu.be/vBMHS6FXBM4?t=325))
- `6:43-10:30` Demo NoMachine install ([https://youtu.be/vBMHS6FXBM4?t=403](https://youtu.be/vBMHS6FXBM4?t=403))
- `10:30-18:41` Demo Xfce4 install ([https://youtu.be/vBMHS6FXBM4?t=630](https://youtu.be/vBMHS6FXBM4?t=630))

### 23.4 Sunshine + Moonlight — **DESCARTADO** para Jetson Nano

**Fuentes:** Issue [moonlight-stream/moonlight-embedded#741](https://github.com/moonlight-stream/moonlight-embedded/issues/741), HN [thread 43439524](https://news.ycombinator.com/item?id=43439524) (2025-03), Stack Overflow #63479215.

**Veredicto:** Sunshine NO funciona en Jetson Nano como servidor host:

1. Sunshine en Linux ARM64 espera **NVFBC** (NVIDIA Frame Buffer Capture) que existe SOLO en GPUs Quadro/GeForce discretas. Tegra X1 no tiene NVFBC.
2. **NVENC en Tegra ≠ NVENC en discrete**: Jetson usa el video engine multimedia de L4T (V4L2) accesible solo via librerías NVIDIA Multimedia API. Sunshine no tiene driver para ese stack.
3. **No hay .deb arm64 para Bionic** en releases recientes de Sunshine (v2026.508.45922 solo distribuye .deb para Ubuntu 22.04+ arm64).
4. Issue #741 abierto desde 2019 sin resolución — sin testimonios de éxito.

**Conclusión:** mantener NoMachine como única solución de escritorio remoto. NO invertir tiempo en intentar Sunshine.

### 23.5 Tailscale en Bionic ARM64 con bug DNS conocido

**Bug crítico verificado:** [Issue tailscale/tailscale#14902](https://github.com/tailscale/tailscale/issues/14902), abierto 2025-02-04 por dmellosanjay. Reproducible en 1.78.1, 1.80.0, 1.82.5, **1.84.0** (verificado por múltiples reporters en 2025). Sin fix oficial en repo.

**Síntoma:** al boot, `tailscaled` arranca y los logs muestran:
```
dns: [rc=unknown ret=direct]
dns: using "direct" mode
logtail: dial "log.tailscale.io:443" failed: no DNS fallback candidates remain
```
Sin DNS, Tailscale no puede contactar el control plane. Workaround empírico: `sudo systemctl restart tailscaled` post-boot funciona; pero el bug puede reaparecer en cualquier link change.

**Causa raíz inferida (comentario @ph1048 sept 2025):** Tailscale modifica `/etc/resolv.conf` al iniciar y si la shutdown anterior no fue limpia, el archivo queda apuntando al resolver interno de Tailscale (`100.100.100.100`) que no es accesible antes de que Tailscale esté up — deadlock circular.

**Fix robusto (NUESTRA configuración) — combinación de tres flags:**

```bash
# Paso 1: instalar Tailscale por repo bionic oficial
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/bionic.noarmor.gpg | \
    sudo tee /usr/share/keyrings/tailscale-archive-keyring.gpg >/dev/null
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/bionic.tailscale-keyring.list | \
    sudo tee /etc/apt/sources.list.d/tailscale.list
sudo apt update && sudo apt install -y tailscale

# Paso 2: prevenir el bug con --accept-dns=false (NO usamos MagicDNS, no necesitamos)
sudo systemctl enable --now tailscaled
sudo tailscale up --accept-dns=false --ssh

# Paso 3: backup defensivo — asegurar que resolv.conf NO es overritten por Tailscale
sudo rm /etc/resolv.conf
sudo ln -sf /run/systemd/resolve/stub-resolv.conf /etc/resolv.conf
sudo systemctl enable --now systemd-resolved
```

**`--accept-dns=false`** previene el bug #14902 por completo porque Tailscale NO toca `/etc/resolv.conf`. Trade-off: pierdes MagicDNS (`nano.tailnet.ts.net` resolverá sólo si la usás directo via IP `100.x.x.x`). **Para nuestro caso es aceptable** — sólo necesitamos el túnel; el nombre lo manejamos con un alias SSH local en el Win11:

```ssh-config
# C:\Users\mitgar14\.ssh\config en Win11
Host nano-tailscale
    HostName 100.78.140.33  # IP fija de Tailscale para la Nano
    User jetson
    IdentityFile ~/.ssh/id_ed25519
```

**`--ssh`** activa **Tailscale SSH** — el daemon de Tailscale acepta conexiones SSH entrantes directamente, así no dependemos del `sshd` del sistema y tenemos auth federada (cualquier device autenticado en el tailnet puede conectarse sin gestionar claves manualmente). **Bonus para demo**: si Win11 también tiene Tailscale, podés conectarte por `ssh nano` directo sin password ni clave (Tailscale verifica identidad).

**WireGuard en kernel 4.9-tegra (recordatorio R6 §9.4):** Tailscale detecta automáticamente que el kernel no tiene módulo WireGuard y usa **userspace networking** (`wireguard-go` embebido). Overhead CPU ~15-25% mayor pero estable. No requiere intervención.

**Headscale en Contabo como backup:** la receta de [mlorente.dev — Headscale Self-Hosted con caso Jetson Nano](https://mlorente.dev/notes/headscale-self-hosted-tailscale/) documenta exactamente nuestro stack (Nano + VPS + Headscale) con el fix definitivo para el deadlock DNS:

```bash
# En la Nano si migramos a Headscale:
sudo systemctl edit systemd-resolved
# Agregar:
[Resolve]
FallbackDNS=1.1.1.1 8.8.8.8

# En /etc/hosts:
echo "<CONTABO_IP> headscale.embebidos3.local" | sudo tee -a /etc/hosts

# Conectar a Headscale en vez de Tailscale Inc.:
sudo tailscale up --login-server=https://headscale.embebidos3.local --accept-dns=false
```

**Decisión operativa:** empezar con Tailscale managed con `--accept-dns=false`. Migrar a Headscale si: (a) free tier de Tailscale tiene downtime durante demo, (b) UAO bloquea controlplane.tailscale.com.

### 23.6 autossh vs alternativas — receta refinada

**Veredicto:** autossh sigue siendo la opción correcta para nuestro caso. Mantener.

**Por qué:**
1. VPS Contabo ya pagado → "single point of failure" argumento NO aplica.
2. autossh + puerto 443 del Contabo es indistinguible de tráfico HTTPS normal — bypass de DPI universitario.
3. Cloudflared free tier: alternativa de backup viable pero requiere cliente `cloudflared` en Win11 además del Tailscale.
4. chisel v1.11.5: superior técnicamente (TLS, fingerprint, backoff built-in) pero requiere instalar y mantener `chisel server` en el VPS Contabo. **Vale el esfuerzo solo post-deadline.**
5. frp: overkill para 1 nodo edge.

**Unit systemd refinada (vs R6 §14):**

```ini
# /etc/systemd/system/autossh-tunnel.service
[Unit]
Description=AutoSSH reverse tunnel Nano -> Contabo
After=network-online.target tailscaled.service
Wants=network-online.target

[Service]
Type=simple
User=jetson
Environment="AUTOSSH_GATETIME=0"
Environment="AUTOSSH_POLL=60"
Environment="AUTOSSH_LOGFILE=/var/log/autossh.log"
ExecStart=/usr/bin/autossh -M 0 -NT \
  -o "ServerAliveInterval=30" \
  -o "ServerAliveCountMax=3" \
  -o "ExitOnForwardFailure=yes" \
  -o "StrictHostKeyChecking=accept-new" \
  -o "UserKnownHostsFile=/home/jetson/.ssh/known_hosts_tunnel" \
  -i /home/jetson/.ssh/id_ed25519 \
  -R 0.0.0.0:2222:localhost:22 \
  -p 443 \
  tunnel@<CONTABO_IP>
Restart=always
RestartSec=15
StartLimitIntervalSec=0

[Install]
WantedBy=multi-user.target
```

**Cambios vs R6 §14:**
- `After=network-online.target tailscaled.service` — espera que Tailscale levante primero por si el VPS está accesible vía tailnet
- `-p 443` — conectar al sshd del Contabo en puerto 443 (configurar Contabo con `Port 443` en `/etc/ssh/sshd_config`)
- `StartLimitIntervalSec=0` — sin límite de restarts (Restart=always estaba siendo overridden por systemd default)
- `UserKnownHostsFile` separado para el túnel — aislamiento de fingerprints
- Usuario dedicado `tunnel@` en Contabo (no `root@`) con `ForceCommand /bin/false` en authorized_keys para que solo pueda hacer port forward

### 23.7 Decisiones nuevas D25-D28

> **D25 — hostapd + driver Wi-Fi USB definitivo (cierre de D16):**
> Para AP mode en TL-WN722N v4 (RTL8188EUS) con Jetson Nano kernel 4.9.337-tegra: usar driver `lwfinger/rtl8188eu` (rama master, fallback `v5.2.2.4`) con hostapd compilado desde `rtl8188eu/hostapd-2.9/` con `CONFIG_DRIVER_RTW=y`. Config: `driver=rtl871xdrv` (NO `nl80211`), `hw_mode=g`, canal 6, WPA2-PSK, `ieee80211n` comentado salvo confirmación de estabilidad. dnsmasq con `dhcp-range=192.168.42.10,192.168.42.50,12h`. Throughput esperado 15-20 Mbps UDP — suficiente para demo. Reemplaza la asunción de R6 de que `aircrack-ng/rtl8188eus` daría AP estable.

> **D26 — Escritorio remoto definitivo (refina D18):**
> NoMachine ARM64 deb + Xfce4 + dummy HDMI plug físico ($5-8 Amazon), `DefaultDesktopCommand "/usr/bin/startxfce4"` en `/usr/NX/etc/node.cfg`. Autologin LightDM al usuario jetson. **Sunshine + Moonlight DESCARTADOS definitivamente** por ausencia de NVFBC en Tegra X1 y falta de .deb arm64 para Bionic. NoMachine es la única opción viable con calidad suficiente para preview OpenCV/TRT.

> **D27 — Tailscale con workaround DNS bug ARM64 (refina D17):**
> Instalar Tailscale 1.84+ desde repo oficial `pkgs.tailscale.com/stable/ubuntu/bionic` (sigue activo en mayo 2026). Activar con `tailscale up --accept-dns=false --ssh` — el `--accept-dns=false` previene el bug #14902 (DNS deadlock en boot); el `--ssh` da auth federada sin gestión manual de claves. En Win11 mapear IP fija Tailscale a alias SSH en `.ssh/config`. **Headscale en Contabo como migration path** si Tailscale Inc. tiene problemas durante demo (receta validada en mlorente.dev caso Jetson + Contabo).

> **D28 — autossh reverse tunnel a Contabo:443 (refina D17, D20):**
> autossh systemd unit con `-p 443` al Contabo (sshd configurado en puerto 443 para indistinguibilidad HTTPS), usuario dedicado `tunnel` con `ForceCommand /bin/false` en authorized_keys, `After=network-online.target tailscaled.service`. **Alternativas no se adoptan ahora**: chisel ARM64 .deb v1.11.5 es backup post-deadline (requiere instalar `chisel server` en Contabo); cloudflared solo si autossh bloqueado por UAO (requiere cliente en Win11).

### 23.8 Runbook integrado v2 — orden de ejecución in-situ

Reemplaza §14 de R6:

```bash
# === FASE 0: Pre-requisitos (asumiendo Driver lwfinger ya instalado per §22.2) ===
# Verificar driver activo
lsmod | grep 8188eu                          # debe mostrar 8188eu cargado
ls /sys/class/ieee80211/                     # debe listar phy

# === FASE 1: Conectar al Wi-Fi remoto vía chip Intel interno (wlan0) ===
nmcli device wifi list
sudo nmcli device wifi connect "<SSID>" password "<PSK>"
ping -c 3 1.1.1.1                            # verificar Internet

# === FASE 2: Xfce4 + NoMachine + autologin ===
sudo apt install -y xfce4 xfce4-goodies xfce4-terminal
sudo tee /etc/lightdm/lightdm.conf.d/12-autologin.conf <<EOF
[Seat:*]
autologin-user=jetson
autologin-user-timeout=0
user-session=xfce
EOF
cd /tmp
wget "https://download.nomachine.com/download/8.16/Arm/nomachine_8.16.1_2_arm64.deb" -O nomachine.deb
sudo dpkg -i nomachine.deb || sudo apt-get install -f -y
grep -q "^DefaultDesktopCommand" /usr/NX/etc/node.cfg || \
  echo 'DefaultDesktopCommand "/usr/bin/startxfce4"' | sudo tee -a /usr/NX/etc/node.cfg
sudo /usr/NX/bin/nxserver --restart

# Conectar dummy HDMI plug físico AHORA
# Reboot:
sudo reboot

# === FASE 3: Tailscale con workaround DNS ===
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/bionic.noarmor.gpg | \
    sudo tee /usr/share/keyrings/tailscale-archive-keyring.gpg >/dev/null
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/bionic.tailscale-keyring.list | \
    sudo tee /etc/apt/sources.list.d/tailscale.list
sudo apt update && sudo apt install -y tailscale
sudo systemctl enable --now tailscaled
sudo rm -f /etc/resolv.conf
sudo ln -sf /run/systemd/resolve/stub-resolv.conf /etc/resolv.conf
sudo systemctl enable --now systemd-resolved
sudo tailscale up --accept-dns=false --ssh
# Seguir URL impresa, autenticar en https://login.tailscale.com

# Verificar
tailscale status
tailscale ip -4                              # tu IP 100.x.x.x — anotar para .ssh/config

# === FASE 4: autossh reverse tunnel a Contabo:443 ===
# Pre-requisito: Contabo VPS ya configurado con:
#   - sshd Port 443 en /etc/ssh/sshd_config
#   - GatewayPorts clientspecified
#   - usuario 'tunnel' con authorized_keys + ForceCommand /bin/false
sudo apt install -y autossh
ssh-keygen -t ed25519 -f /home/jetson/.ssh/id_ed25519_tunnel -N ""
ssh-copy-id -i ~/.ssh/id_ed25519_tunnel.pub -p 443 tunnel@<CONTABO_IP>

sudo tee /etc/systemd/system/autossh-tunnel.service <<'EOF'
[Unit]
Description=AutoSSH reverse tunnel Nano -> Contabo
After=network-online.target tailscaled.service
Wants=network-online.target

[Service]
Type=simple
User=jetson
Environment="AUTOSSH_GATETIME=0"
Environment="AUTOSSH_POLL=60"
ExecStart=/usr/bin/autossh -M 0 -NT \
  -o "ServerAliveInterval=30" \
  -o "ServerAliveCountMax=3" \
  -o "ExitOnForwardFailure=yes" \
  -o "StrictHostKeyChecking=accept-new" \
  -i /home/jetson/.ssh/id_ed25519_tunnel \
  -R 0.0.0.0:2222:localhost:22 \
  -p 443 \
  tunnel@<CONTABO_IP>
Restart=always
RestartSec=15
StartLimitIntervalSec=0

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now autossh-tunnel

# Verificar desde Win11: ssh jetson@<CONTABO_IP> -p 2222

# === FASE 5: AP mode standalone (sólo para demo battery-powered) ===
# NO ejecutar en dev — sólo si llegamos a escenario sin Wi-Fi conocida
cd ~/rtl8188eu/hostapd-2.9/hostapd
cp defconfig .config
echo "CONFIG_DRIVER_RTW=y" >> .config
make -j4
sudo cp hostapd /usr/local/sbin/hostapd-rtw

sudo apt install -y dnsmasq
sudo tee /etc/dnsmasq.d/embebidos3-ap.conf <<'EOF'
interface=wlan1
bind-interfaces
dhcp-range=192.168.42.10,192.168.42.50,12h
dhcp-option=3,192.168.42.1
domain-needed
bogus-priv
no-resolv
server=1.1.1.1
EOF

sudo tee /etc/hostapd/hostapd-rtw.conf <<'EOF'
interface=wlan1
ctrl_interface=/var/run/hostapd
ssid=embebidos3-nano
channel=6
wpa=2
wpa_passphrase=embebidos3demo2026
driver=rtl871xdrv
hw_mode=g
wpa_key_mgmt=WPA-PSK
wpa_pairwise=CCMP
rsn_pairwise=CCMP
max_num_sta=4
EOF

# Script para activar AP on-demand:
sudo tee /usr/local/sbin/embebidos3-ap-up.sh <<'EOF'
#!/bin/bash
sudo nmcli device set wlan1 managed no
sudo ip addr add 192.168.42.1/24 dev wlan1
sudo ip link set wlan1 up
sudo /usr/local/sbin/hostapd-rtw -B /etc/hostapd/hostapd-rtw.conf
sudo systemctl restart dnsmasq
EOF
sudo chmod +x /usr/local/sbin/embebidos3-ap-up.sh

# Para activar en demo:
# sudo /usr/local/sbin/embebidos3-ap-up.sh
```

### 23.9 Fuentes Ronda 8 (62-78)

| # | Título | URL | Tipo | Fecha |
|---|---|---|---|---|
| 62 | lwfinger/rtl8188eu issue #363 — AP mode Jetson Nano TL-WN722N V3 v5.2.2.4 | https://github.com/lwfinger/rtl8188eu/issues/363 | Issue resuelto | 2021-03-02 |
| 63 | lwfinger/rtl8188eu issue #257 — Problems with hostapd | https://github.com/lwfinger/rtl8188eu/issues/257 | Issue | 2017+ |
| 64 | lwfinger/rtl8188eu issue #3 — Can not build Soft-AP using hostapd | https://github.com/lwfinger/rtl8188eu/issues/3 | Issue (referencia histórica) | 2014+ |
| 65 | lwfinger/rtl8188eu issue #94 — hostapd TL-WN725N v2 kernel 3.10 | https://github.com/lwfinger/rtl8188eu/issues/94 | Issue | 2016 |
| 66 | lwfinger/rtl8188eu rtl_hostapd.conf master | https://github.com/lwfinger/rtl8188eu/blob/master/rtl_hostapd.conf | Config oficial | Vigente 2026 |
| 67 | LWN.net — wifi: rtl8xxxu Add AP mode support for 8188f (Martin Kaistra) | https://lwn.net/Articles/930394/ | LWN article | 2023 |
| 68 | NoMachine KB AR02R01074 — Tips Jetson Nano (updated 2025-07-03) | https://kb.nomachine.com/AR02R01074 | Doc oficial NoMachine | 2025-07 |
| 69 | NoMachine forum — headless HDMI signal (Jetson Orin Nano caso) | https://forum.nomachine.com/topic/how-can-i-get-a-local-hdmi-signal-if-headless | Foro oficial | 2024-02 |
| 70 | Amplifi Labs — NoMachine + XFCE on Headless VPS | https://www.amplifilabs.com/post/nomachine-with-xfce-desktop-on-headless-vps-complete-setup-guide | Blog técnico | 2026-03-30 |
| 71 | make2explore Systems — NoMachine on Jetson Nano (video) | https://youtu.be/vBMHS6FXBM4 | Video tutorial | 2024-10-18 |
| 72 | luohanjie.com — NoMachine + Xfce4 en Jetson Nano (Chinese) | http://luohanjie.com/2020-09-10/using-nomachine-on-jetson-nano-with-xfce.html | Blog técnico | 2020-09 |
| 73 | NVIDIA Forum — Auto switching physical display and dummy Xorg | https://forums.developer.nvidia.com/t/auto-switching-between-physical-display-and-dummy-xorg-for-headless-jetson-orin-nano/364083 | Foro NVIDIA | 2026-03-19 |
| 74 | moonlight-stream/moonlight-embedded issue #741 — Jetson Nano sin NVENC | https://github.com/moonlight-stream/moonlight-embedded/issues/741 | Issue abierto sin fix | 2019-2021 |
| 75 | tailscale/tailscale issue #14902 — DNS bug ARM64 1.78-1.84+ | https://github.com/tailscale/tailscale/issues/14902 | Issue abierto | 2025-02 |
| 76 | Tailscale docs — Install on Ubuntu 18.04 bionic | https://tailscale.com/kb/1037/install-ubuntu-1804 | Doc oficial | 2025-02 verificado |
| 77 | mlorente.dev — Headscale Self-Hosted (caso Jetson Nano + Contabo) | https://mlorente.dev/notes/headscale-self-hosted-tailscale/ | Blog técnico | 2025 |
| 78 | jpillora/chisel v1.11.5 release (arm64 .deb disponible) | https://github.com/jpillora/chisel/releases/tag/v1.11.5 | Release | 2026-03-09 |

### 23.10 Gaps de evidencia residuales R8

1. **Estabilidad de hostapd + driver lwfinger master vs v5.2.2.4 en kernel 4.9.337-tegra específico**: no hay testimonio directo (#363 fue en kernel 4.9.140 de JP 4.5). Mitigación: probar primero master, fallback a v5.2.2.4 si hay desconexiones.
2. **¿UAO bloquea controlplane.tailscale.com?**: no hay evidencia directa. Mitigación: tener Headscale en Contabo listo como migration path.
3. **¿UAO bloquea SSH al puerto 443 del Contabo?**: improbable (es indistinguible de HTTPS) pero no verificable hasta demo. Mitigación: cloudflared como tercer fallback (no implementado en runbook v2).
4. **Performance Tailscale userspace networking en kernel 4.9 con inferencia activa**: estimado 15-25% overhead CPU; medible empíricamente con `top` o `tegrastats` durante inferencia.

### 23.11 Historial actualizado

| Ronda | Fecha | Profundidad | Foco |
|---|---|---|---|
| 6 | 2026-05-12 | Alto | Acceso remoto Win11 → Nano + TL-WN722N v4 + Contabo bastion + overlay networking |
| 7 | 2026-05-12 | Bajo | Fix DKMS install RTL8188EUS — corrección kernel version + path getKernelSources + sintaxis `--kernelsourcedir` |
| 7-bis | 2026-05-12 | In-situ | Cierre empírico — driver real, USB ID real, WiFi Intel descubierto (§22, D21-D24) |
| **8** | **2026-05-12** | **Media** | **Implementación final — hostapd lwfinger, NoMachine Xfce4, Tailscale Bionic bug DNS, autossh refinado (§23, D25-D28)** |

---

**Fin de la Ronda 8.** Stack acceso remoto fully spec'd. Próxima acción in-situ: ejecutar runbook §23.8 secuencial fases 1-4 (fase 5 sólo para demo). Para ampliaciones futuras (más nodos edge, fleet management) considerar Ronda 9 sobre Headscale + Docker compose.

**Fin Ronda 7.**
