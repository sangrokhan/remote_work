#!/usr/bin/env bash
# Milvus named volume backup / restore
# Usage:
#   backup:  ./scripts/milvus_volume.sh backup [OUTPUT_DIR]
#   restore: ./scripts/milvus_volume.sh restore <BACKUP_DIR>

set -euo pipefail

COMPOSE_FILE="configs/milvus/docker-compose.yml"
VOLUME_NAMES=("etcd_data" "minio_data" "milvus_data")

usage() {
  echo "Usage:"
  echo "  $0 backup  [output_dir]   — backup volumes to tar.gz files"
  echo "  $0 restore <backup_dir>   — restore volumes from tar.gz files"
  exit 1
}

resolve_volume() {
  local short_name="$1"
  # docker volume ls로 실제 볼륨 이름 탐색 (prefix 자동 감지)
  docker volume ls --format '{{.Name}}' | grep -E "_${short_name}$" | head -1
}

do_backup() {
  local output_dir="${1:-./milvus_backup_$(date +%Y%m%d_%H%M%S)}"
  mkdir -p "$output_dir"
  echo "[backup] → $output_dir"

  for short_name in "${VOLUME_NAMES[@]}"; do
    local vol
    vol=$(resolve_volume "$short_name")
    if [[ -z "$vol" ]]; then
      echo "[SKIP] volume matching *_${short_name} not found"
      continue
    fi
    local out_file="$output_dir/${short_name}.tar.gz"
    echo "  $vol → $out_file"
    docker run --rm \
      -v "${vol}:/data:ro" \
      -v "$(realpath "$output_dir"):/backup" \
      alpine tar czf "/backup/${short_name}.tar.gz" -C /data .
  done

  echo "[backup] done: $(realpath "$output_dir")"
}

do_restore() {
  local backup_dir="${1:-}"
  if [[ -z "$backup_dir" || ! -d "$backup_dir" ]]; then
    echo "Error: backup_dir required and must exist"
    usage
  fi

  echo "[restore] ← $backup_dir"
  echo "WARNING: This will overwrite existing volume data."
  read -r -p "Continue? [y/N] " confirm
  [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }

  # 기존 컨테이너 중지
  docker compose -f "$COMPOSE_FILE" down 2>/dev/null || true

  for short_name in "${VOLUME_NAMES[@]}"; do
    local tar_file="$backup_dir/${short_name}.tar.gz"
    if [[ ! -f "$tar_file" ]]; then
      echo "[SKIP] $tar_file not found"
      continue
    fi

    local vol
    vol=$(resolve_volume "$short_name")
    if [[ -z "$vol" ]]; then
      # 볼륨이 없으면 compose 프로젝트명 추론해서 생성
      local project
      project=$(basename "$(dirname "$(realpath "$COMPOSE_FILE")")")
      vol="${project}_${short_name}"
      echo "  creating volume: $vol"
      docker volume create "$vol" > /dev/null
    fi

    echo "  $tar_file → $vol"
    docker run --rm \
      -v "${vol}:/data" \
      -v "$(realpath "$backup_dir"):/backup:ro" \
      alpine sh -c "rm -rf /data/* /data/..?* /data/.[!.]* 2>/dev/null; tar xzf /backup/${short_name}.tar.gz -C /data"
  done

  echo "[restore] done — run: docker compose -f $COMPOSE_FILE up -d"
}

case "${1:-}" in
  backup)  do_backup  "${2:-}" ;;
  restore) do_restore "${2:-}" ;;
  *)       usage ;;
esac
