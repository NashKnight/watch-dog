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

参数在 `config.json` 改:`interval`(检查间隔)、`report_interval`(播报间隔,测试期 30、正式改 600)、`stall`(多久无更新算卡死)。改后 `restart`。

## 原理

### 架构:一个守护进程 + 一张监控表

`start` 拉起一个常驻进程,每 `interval` 秒重读 `watchlist.txt`,对表里每个 job 生成一份统一的 `JobReport`,再由通知状态机决定要不要发飞书。改表 / 改配置无需重启(改配置需 `restart`)。

### 数据从哪来(不必登训练节点)

所有信号都来自**共享盘**上的文件,所以在登录 / notebook 节点即可监控别的节点上跑的 job:

- **训练任务**:按 job id 在 `search_roots` 下找到 `all_config.json`(训练会把它写到 `exp_ckpt_dir`),从中读出 `tensorboard` 路径、`max_steps`、`warmup_t`、`save_step`、`world_size`(=GPU 卡数)。
  再**增量解析 TensorBoard event 文件**(TFRecord 格式,记录已读字节偏移,每轮只读新增部分,避免文件涨到几百 MB 后全量重读),拿到当前 `step`、各 `loss`、`wall_time`。
  - 进度% = `step / max_steps`
  - 速度 = 采样窗口内 `Δstep / Δwall`,据此算 ETA 与预计完成时刻
  - 阶段 = 依据 `step` 与 `warmup_t` 判断 warmup / 正式训练,并提示下一次存 ckpt 的 step
  - 新鲜度 = `now - event文件mtime`
- **通用(非训练)任务**:给一个日志路径(`--log`,或用 `--cmd` 从整条启动命令里自动抽 `>>`/`tee` 的重定向目标并用其中的 `cd` 拼成绝对路径)。用日志 `mtime` 判新鲜度,并用正则从日志尾部抓 `X/Y` 当进度。
- **可选 `ct`**:本机若装了 Cybertron 的 `ct`(`config.json` 里 `use_ct: true`),用它拿**权威状态**(Running/Failed/Killed/Succeeded)与 **CPU/GPU/内存/资源池**,对任意任务类型都生效。没有 `ct` 时,状态靠"日志/TB 是否还在更新"来推断。

### 状态判定

`WarmingUp`(启动宽限期内还没数据) · `Running` · `Stalled`(超过 `stall` 秒无更新,疑似卡死/被杀) · `Completed`(step ≥ max_steps) · `Failed`/`Killed`(来自 ct) · `Unknown`(既非训练、又没日志、也没 ct)。

### 通知状态机(每个 job 独立)

- 首次纳入监控:发一次「已开始监控」。
- 进入异常(Stalled/Failed/Killed):立即发**红色告警**;同一异常只发一次,不刷屏。
- 从异常恢复:发**绿色恢复**。
- 正常运行:每 `report_interval` 秒发一次**蓝色进度**。
- 训练完成:发**绿色完成**。

每个 job 的通知/进度状态存在 `state/<id>.json`;全表快照写 `status.md`;运行日志 `watchdog.log`;事件流 `events.jsonl`。

### 目录

```
watchdog.py   # 守护进程 + CLI + 统一 JobReport + 状态机
tb_reader.py  # TensorBoard event 增量解析
notifier.py   # 飞书卡片
config.json   watchlist.txt   jobs_extra.json   state/   status.md   （运行时产物, 已 gitignore）
```
