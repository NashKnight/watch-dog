# stage1_dialogue_generation

用于监控 omnipro 数据合成里“Stage1 对话生成 + 自动质检/修订 + materialize/viz”的日志。

## 适用日志

适用于 `counting`、`step_inst`、`narration`、`alert` 等流水线中类似下面的日志结构：

```text
[prepare] ...
[stage0] ...
[round 1/12] have 0/50, generating 400 window(s)
Stage 1: generate ...
Stage 1 verify ...
[refine] ...
[fill] accepted 50 new ...
PIPELINE DONE
```

## 监控口径

- 进度：只读取最新的 `have X/target`，表示最终已接受样本数，不读取单轮 `accepted=A/B`。
- 阶段：自动显示 `prepare 准备标注`、`stage0 切片规划`、`第R/N轮 · 生成中`、`第R/N轮 · 校验中`、`已达标, 收尾(materialize/viz)`、`完成`。
- 完成：只认 `PIPELINE DONE`，即使 `have X/target` 达标，也要等 materialize/viz 完成。
- ETA：日志没有行级时间戳时，用 watchdog 自己的轮询历史估算速度和剩余时间。

## 添加任务

```bash
cd /user/hongchenye/watchdog && PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python watchdog.py add <job_id> <task_name> --preset stage1_dialogue_generation --log <absolute_log_path>
```

示例：

```bash
cd /user/hongchenye/watchdog && PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python watchdog.py add 144258 narration_stage1 --preset stage1_dialogue_generation --log /user/hongchenye/omni_duplex_synthetic_data/omnipro_data_construction/narration/logs/en_stage1_50_tts_ready.log
```
