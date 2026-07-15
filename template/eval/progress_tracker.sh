#!/bin/bash
# 评测旁路进度：统计 result.json，写标准 X/Y 到 progress.log。
# 用法: progress_tracker.sh <OUTPUT_DIR> <TOTAL> [INTERVAL_SEC]
set -u
O="$1"
TOTAL="${2:-1986}"
INT="${3:-30}"
LOG="$O/progress.log"
mkdir -p "$O"
echo "$(date '+%F %T') tracker start: dir=$O total=$TOTAL interval=${INT}s" >> "$LOG"
while true; do
  n=$(find "$O" -name result.json 2>/dev/null | wc -l)
  echo "$(date '+%F %T') eval progress: ${n}/${TOTAL}" >> "$LOG"
  # 让 mtime 反映真实 shard 活动；无 shard 日志时保留心跳 mtime 供启动宽限期使用。
  newest=$(find "$O" -name '*.log' ! -name "$(basename "$LOG")" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
  [ -n "$newest" ] && touch -r "$newest" "$LOG" 2>/dev/null || true
  if [ "$n" -ge "$TOTAL" ]; then
    echo "$(date '+%F %T') eval done: ${n}/${TOTAL}" >> "$LOG"
    break
  fi
  sleep "$INT"
done
