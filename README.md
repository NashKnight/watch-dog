# Train Watchdog

一个常驻进程读一张监控表(`watchlist.txt`,每行 `<job_id> <名字>`),自动监控每个 job 的状态与进度,异常/进度都发飞书。播报标题为 `watchdog播报：任务 <id>（<名字>）`。

## Quickstart

```bash
source /root/minicpmo/bin/activate && cd /user/hongchenye/watchdog

# 0. 配 webhook: 编辑 config.json 的 feishu_webhook, 然后验证
python watchdog.py test-notify

# 1. 加任务(只认 job id)
python watchdog.py add 141369 stage2训练          # 训练任务: 自动找 TB/配置
python watchdog.py add 141394 数据生成 --cmd "整条启动命令"   # 非训练: 从命令抽日志路径

# 2. 开 / 关
python watchdog.py start          # 开启守护进程
python watchdog.py stop           # 关闭
python watchdog.py restart        # 改完 config.json 后重启生效

# 3. 查
python watchdog.py ls             # 监控表 + 状态/进度
python watchdog.py show 141369    # 某 job 详细快照
python watchdog.py rm 141369      # 移除
```

## 说明

- 参数在 `config.json` 改:`interval`(检查间隔)、`report_interval`(播报间隔,测试期 30、正式改 600)、`stall`(多久无更新算卡死)。改后 `restart`。
- 训练任务字段最全(进度/速度/ETA/loss/GPU 卡数);非训练任务给日志(`--cmd` 或 `--log`)后按日志新鲜度+进度监控。
- CPU/内存、平台级权威状态(Failed/Killed)需本机有 `ct`,`config.json` 里 `use_ct: true` 才能点亮。
