# 推理与评测模板

当评测按样本落 `result.json` 时，启动本目录的旁路 tracker：

```bash
bash template/eval/progress_tracker.sh <输出目录> <总样本数>
```

它向 `<输出目录>/progress.log` 写标准进度和完成标记。再将该文件配置为通用任务日志：

```bash
python watchdog.py add <job_id> <评测名> --log <输出目录>/progress.log
```

默认按递归 `result.json` 数量统计；如果评测落盘结构不同，请复制到 `template/temp/` 后定制，不要在根目录增加任务专用解析器。
