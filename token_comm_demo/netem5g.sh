#!/bin/bash
IFACE=enp1s0f0   # 단말로 향하는 다운링크 인터페이스로 교체

case "$1" in
  on)  sudo tc qdisc add dev $IFACE root netem rate 60mbit loss 1%
       echo "5G busy ON (60Mbps, loss 1%)" ;;
  off) sudo tc qdisc del dev $IFACE root
       echo "OFF" ;;
  status) sudo tc qdisc show dev $IFACE ;;
  *)   echo "사용법: $0 {on|off|status}" ;;
esac
