#!/bin/bash
# 旁路进度统计: 周期性数 humanevalkit 评测已完成的 result.json 数量,
# 写成 "X/Y" 格式(带时间戳)到 progress.log, 供 watchdog 抓进度条 + 判活 + 估 ETA。
# 用法: eval_progress_tracker.sh <OUTPUT_DIR> <TOTAL> [INTERVAL_SEC]
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
  if [ "$n" -ge "$TOTAL" ]; then
    echo "$(date '+%F %T') eval done: ${n}/${TOTAL}" >> "$LOG"
    break
  fi
  sleep "$INT"
done
