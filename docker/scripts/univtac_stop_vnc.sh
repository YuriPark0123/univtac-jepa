#!/bin/bash
# VNC 데스크탑 종료.  사용: stop_vnc.sh   (VNC_DISPLAY 기본 1)
N="${VNC_DISPLAY:-1}"
vncserver -kill ":${N}" 2>/dev/null && echo "VNC :${N} (포트 $((5900 + N))) 종료" || echo "VNC :${N} 이(가) 떠 있지 않습니다"
rm -f "/tmp/.X${N}-lock" "/tmp/.X11-unix/X${N}" 2>/dev/null || true
