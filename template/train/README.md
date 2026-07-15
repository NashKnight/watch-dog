# 模型训练模板

标准训练任务直接由根目录的 `watchdog.py` 自动发现 TensorBoard：

```bash
cd /user/hongchenye/watchdog
python watchdog.py add <job_id> <任务名>
```

自动发现依赖训练输出目录内的 `all_config.json`，其中应包含：

- `training_args.tensorboard`：TensorBoard event 目录；
- `training_args.max_steps`：总 step；
- 可选 `warmup_t`、`save_step`、`world_size`、`exp_ckpt_dir`。

watchdog 会从 event 文件增量读取当前 step、loss、速度和 ETA。训练不需要为每个任务新写日志解析脚本；只有无法落 TensorBoard 的训练，才应在 `template/temp/` 定制旁路 tracker。
