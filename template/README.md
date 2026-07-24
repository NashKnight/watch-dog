# Watchdog 模板

根目录只保留通用监控核心。需要匹配特定产物或日志格式时，从这里选择模板。

- `train/`：TensorBoard 模型训练。核心会按 job id 自动发现 `all_config.json` 和 event 文件，通常不需要旁路脚本。
- `eval/`：推理/评测。`progress_tracker.sh` 统计 `result.json` 并输出通用 `X/Y` 进度日志。
- `stage1_dialogue_generation/`：omnipro Stage1 对话生成/质检/修订流水线。使用 `--preset stage1_dialogue_generation`，显示最终通过进度和当前阶段。
- `stage2_tts_generation/`：omnipro Stage2 TTS。使用 `--preset stage2_tts_generation`，按 clip 合成和 sample 写盘两段显示进度。
- `temp/`：SWW、TTS、一次性 hook 等任务专用模板。它不进入 Git；使用前复制并按具体路径、完成标记和进度口径定制。

## 通用日志契约

给 `jobs_extra.json` 配置的日志应满足：

1. 每条进度记录只有一个可解析的 `X/Y`，并带 `YYYY-MM-DD HH:MM:SS` 时间戳；
2. 完成时写入 `eval done:`、`all done:`、`gen done:` 或 `tts done:`；
3. 日志 mtime 应反映真实业务写盘，而不是仅由 tracker 定时刷新。

若一行必须包含分支明细，请把明细写成 `done of total`，避免被通用 `X/Y` 解析器误识别为总进度。
