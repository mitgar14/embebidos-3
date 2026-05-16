#!/usr/bin/env bash
# Instala units systemd + sudoers + secrets template + wrapper + tmpfiles.
# Idempotente. Ejecutar como user jetson (que invoca sudo internamente).
set -euo pipefail

ROOT=/home/jetson/embebidos-3
REPO_SYSTEMD="${ROOT}/systemd"

echo "[1/8] crear /etc/embebidos3/"
sudo mkdir -p /etc/embebidos3
if [[ ! -f /etc/embebidos3/secrets.env ]]; then
    sudo tee /etc/embebidos3/secrets.env > /dev/null <<'EOF'
# HF_TOKEN requerido para descargar repo privado mitgar14/embebidos-3-models
HF_TOKEN=hf_REEMPLAZAR
EMBEBIDOS3_TRTEXEC_WORKSPACE=512
EOF
    sudo chown root:jetson /etc/embebidos3/secrets.env
    sudo chmod 0640 /etc/embebidos3/secrets.env
    echo "    creado, editá con 'sudo nano /etc/embebidos3/secrets.env' antes de continuar"
fi

echo "[2/8] instalar wrapper sudoers-safe"
sudo install -m 0755 "${ROOT}/scripts/embebidos3-builder-launch" /usr/local/bin/embebidos3-builder-launch

echo "[3/8] instalar sudoers (14 entradas)"
sudo install -m 0440 -o root -g root /dev/stdin /etc/sudoers.d/embebidos3 <<'EOF'
jetson ALL=(root) NOPASSWD: /usr/local/bin/embebidos3-builder-launch *
jetson ALL=(root) NOPASSWD: /bin/systemctl stop embebidos3-server.service
jetson ALL=(root) NOPASSWD: /bin/systemctl start embebidos3-server.service
jetson ALL=(root) NOPASSWD: /bin/systemctl stop embebidos3-builder@*.service
jetson ALL=(root) NOPASSWD: /bin/systemctl stop lightdm.service
jetson ALL=(root) NOPASSWD: /bin/systemctl start lightdm.service
jetson ALL=(root) NOPASSWD: /bin/systemctl disable nvzramconfig
jetson ALL=(root) NOPASSWD: /sbin/swapoff -a
jetson ALL=(root) NOPASSWD: /sbin/swapon /mnt/swap.img
jetson ALL=(root) NOPASSWD: /sbin/sysctl vm.swappiness=*
jetson ALL=(root) NOPASSWD: /usr/bin/fallocate -l 8G /mnt/swap.img
jetson ALL=(root) NOPASSWD: /bin/chmod 600 /mnt/swap.img
jetson ALL=(root) NOPASSWD: /sbin/mkswap /mnt/swap.img
jetson ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches
EOF
sudo visudo -cf /etc/sudoers.d/embebidos3

echo "[4/8] instalar units systemd"
sudo install -m 0644 "${REPO_SYSTEMD}/embebidos3-server.service" /etc/systemd/system/
sudo install -m 0644 "${REPO_SYSTEMD}/embebidos3-builder@.service" /etc/systemd/system/

echo "[5/8] instalar tmpfiles config (logs TTL 3d)"
sudo install -m 0644 "${REPO_SYSTEMD}/embebidos3-logs.tmpfiles.conf" /etc/tmpfiles.d/

echo "[6/8] systemctl daemon-reload + enable + tmpfiles --create"
sudo systemctl daemon-reload
sudo systemctl enable embebidos3-server.service
sudo systemd-tmpfiles --create /etc/tmpfiles.d/embebidos3-logs.tmpfiles.conf

echo "[7/8] migrar server actual (si está corriendo con nohup, no systemd)"
if ! systemctl is-active --quiet embebidos3-server.service \
   && pgrep -f 'nano_server:app' > /dev/null; then
    echo "    matando server nohup..."
    pkill -f 'nano_server:app' || true
    sleep 2
fi

echo "[8/8] start server vía systemd"
sudo systemctl start embebidos3-server.service
sleep 3
systemctl status embebidos3-server.service --no-pager | head -15

cat <<'EOF'

==============================================================================
  Setup completo
==============================================================================
  Verificá:
    curl http://localhost:8000/health
    curl http://localhost:8000/model/state
    systemctl status embebidos3-server.service
    sudo -l -U jetson | grep embebidos3

  HF_TOKEN: editá /etc/embebidos3/secrets.env con el token real.
  Después: sudo systemctl restart embebidos3-server.service
==============================================================================
EOF
