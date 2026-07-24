# OmniPro Stage2 TTS 监控模板

用于 `omnipro_data_construction/counting/generate_audio.py` 这一类 Stage2 TTS 任务。

## 使用方式

```bash
cd /user/hongchenye/watchdog && \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python watchdog.py add <job_id> <task_name> \
  --preset stage2_tts_generation \
  --log <absolute_stage2_tts_log_path> && \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python watchdog.py start
```

兼容旧名字：`--preset tts_audio_generation` 和 `--preset tts`。

## 日志契约

Stage2 TTS 进度分两段展示，避免等到整条音轨最终落盘才更新：

1. clip 合成阶段：
   - 总数：`[tts] clip-total <N> (chattts=<A> cosyvoice=<B>)`
   - 每完成或命中缓存一个 clip：`[tts-clip] + ...` 或 `[tts-clip] . ... cached`
   - watchdog 进度条显示 `已完成 clip / 总 clip`
2. sample 写盘阶段：
   - 总数：`[tts] sample-total <N>`
   - 每写完一条样本的 user/ai wav 并回写 JSON：`[tts]  + <uid> ...`
   - watchdog 进度条显示 `已写盘 sample / 总 sample`
3. 完成：
   - 纯 TTS：`[tts] done: ...`
   - 如果 entry 在 TTS 后还串联了可视化，给该 job 覆盖 `done_regex`，让完成标记落在最后一步，例如：

```json
"done_regex": "\\[viz\\] wrote .*conditional_alert/demo\\.html"
```

## 卡住判断

watchdog 卡片会显示：

- `进度`：当前 clip 或 sample 的进度条。
- `阶段`：准备样本、合成 ChatTTS/CosyVoice clips、写入整段 user/ai 音轨、完成可视化等。
- `数据更新`：距离上次日志写入过去多久。
- `最后日志写入`：最后一次业务日志写入的精确时间。

`generate_audio.py` 的关键进度日志必须 `flush=True`，否则管道/`tee` 可能缓冲，导致 Feishu 看到的进度滞后。
