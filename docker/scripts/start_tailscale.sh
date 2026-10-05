#!/bin/bash
# -----------------------------------------------------------------------------
# Tailscale(VPN) 자동 연결 스크립트
#  - /dev/net/tun + NET_ADMIN 이 있으면 → 커널 모드(완전한 VPN, ping/SSH 투명 라우팅)
#  - 없으면 → 자동으로 유저스페이스 모드(SOCKS5 프록시 localhost:1055)로 폴백
#  - 환경변수 TS_AUTHKEY 가 있으면 자동 로그인, 없으면 로그인 URL 만 출력
# -----------------------------------------------------------------------------
mkdir -p /var/run/tailscale /var/lib/tailscale

# 이미 데몬이 떠 있으면 중복 실행하지 않음
if ! pidof tailscaled >/dev/null 2>&1; then

    TUN_ARGS=""
    if [ ! -c /dev/net/tun ]; then
        echo "[tailscale] /dev/net/tun 이 없어 유저스페이스(SOCKS5 localhost:1055) 모드로 실행합니다."
        TUN_ARGS="--tun=userspace-networking --socks5-server=localhost:1055"
    fi

    tailscaled \
        ${TUN_ARGS} \
        --state=/var/lib/tailscale/tailscaled.state \
        --socket=/var/run/tailscale/tailscaled.sock \
        > /var/log/tailscaled.log 2>&1 &

    # 데몬 소켓이 준비될 때까지 잠깐 대기
    for i in $(seq 1 20); do
        [ -S /var/run/tailscale/tailscaled.sock ] && break
        sleep 0.5
    done
fi

TS_HOSTNAME="${TS_HOSTNAME:-$(hostname)}"

if [ -n "${TS_AUTHKEY}" ]; then
    echo "[tailscale] auth key 로 자동 로그인 중..."
    tailscale up --authkey="${TS_AUTHKEY}" --hostname="${TS_HOSTNAME}" ${TS_EXTRA_ARGS}
    echo "[tailscale] 연결 완료. 이 노드의 tailnet IP: $(tailscale ip -4 2>/dev/null | head -n1)"
else
    echo "[tailscale] TS_AUTHKEY 가 없습니다. 컨테이너 안에서 다음을 직접 실행해 로그인하세요:"
    echo "            tailscale up"
fi
