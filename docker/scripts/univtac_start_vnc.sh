#!/bin/bash
# -----------------------------------------------------------------------------
# TigerVNC 데스크탑 시작 스크립트 (기존 scripts/start_vnc.sh 와 동일, display 번호만 변수화)
#   VNC_DISPLAY (기본 1) → 포트 5900+N.  다른 컨테이너의 VNC 와 겹치면 VNC_DISPLAY=2 로.
# -----------------------------------------------------------------------------
set -e

VNC_PASSWORD="${VNC_PASSWORD:-isaaclab}"
GEOMETRY="${VNC_GEOMETRY:-1920x1080}"
N="${VNC_DISPLAY:-1}"

mkdir -p ~/.vnc

# 1) 비밀번호 파일 생성 (없을 때만)
VNCPASSWD_BIN="$(command -v vncpasswd || command -v tigervncpasswd)"
if [ ! -f ~/.vnc/passwd ]; then
    if [ -z "${VNCPASSWD_BIN}" ]; then
        echo "[에러] vncpasswd/tigervncpasswd 가 없습니다. (apt 로 tigervnc-tools 설치 필요)"
        exit 1
    fi
    echo -e "${VNC_PASSWORD}\n${VNC_PASSWORD}\nn" | "${VNCPASSWD_BIN}"
fi

# 2) xstartup (XFCE 데스크탑 실행)
cat > ~/.vnc/xstartup << 'XEOF'
#!/bin/bash
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
export XKL_XMODMAP_DISABLE=1
exec startxfce4
XEOF
chmod +x ~/.vnc/xstartup

# 3) XFCE 절전/화면꺼짐 비활성화
mkdir -p ~/.config/xfce4/xfconf/xfce-perchannel-xml/
cat > ~/.config/xfce4/xfconf/xfce-perchannel-xml/xfce4-power-manager.xml << 'PMEOF'
<?xml version="1.0" encoding="UTF-8"?>
<channel name="xfce4-power-manager" version="1.0">
  <property name="xfce4-power-manager" type="empty">
    <property name="show-tray-icon" type="bool" value="false"/>
    <property name="blank-on-ac" type="int" value="0"/>
    <property name="dpms-on-ac-off" type="int" value="0"/>
    <property name="dpms-on-ac-sleep" type="int" value="0"/>
    <property name="dpms-enabled" type="bool" value="false"/>
  </property>
</channel>
PMEOF
cat > ~/.config/xfce4/xfconf/xfce-perchannel-xml/xfce4-screensaver.xml << 'SSEOF'
<?xml version="1.0" encoding="UTF-8"?>
<channel name="xfce4-screensaver" version="1.0">
  <property name="saver" type="empty">
    <property name="enabled" type="bool" value="false"/>
    <property name="idle-activation" type="empty">
      <property name="enabled" type="bool" value="false"/>
    </property>
  </property>
  <property name="lock" type="empty">
    <property name="enabled" type="bool" value="false"/>
  </property>
</channel>
SSEOF

# 4) 기존 세션 정리 후 새로 시작
vncserver -kill ":${N}" >/dev/null 2>&1 || true
rm -f "/tmp/.X${N}-lock" "/tmp/.X11-unix/X${N}" 2>/dev/null || true

# -localhost no : host 네트워크에서 SSH 터널로 접속하기 위함 (외부 직접 노출 금지, SSH 터널만)
vncserver ":${N}" -geometry "${GEOMETRY}" -depth 24 -localhost no

# 5) 화면 꺼짐 방지 (런타임)
export DISPLAY=":${N}"
xset s off 2>/dev/null || true
xset s noblank 2>/dev/null || true
xset -dpms 2>/dev/null || true

echo ""
echo "========================================================"
echo " VNC 데스크탑이 :${N} (포트 $((5900 + N))) 에서 실행 중입니다."
echo " 비밀번호: ${VNC_PASSWORD}"
echo " 해상도: ${GEOMETRY}"
echo "========================================================"
