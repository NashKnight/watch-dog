# Train Watchdog · 任务看门狗

一个常驻**守护进程**读取一张**监控表**，自动监控每个 job 的**状态**与**进度**，并**自动发飞书播报**，省得自己反复翻日志。

## 心智模型：一个开关 + 一张表

- **监控表** = 纯文本 `watchlist.txt`，每行一个 job：`<job_id> <任务名>`（名字可选）。
- **守护进程** = 一个进程，`start` 开 / `stop` 关。它每隔一段时间重读监控表，逐个 job 评估、按需播报。
- 增删查改都只认 **job id**：`add` / `rm` / `ls` / `show`。改了表，守护进程下一轮自动生效。

## 功能

1. **状态异常立即告警（红卡）**：训练卡死（TB 长时间无更新）/ `Failed` / `Killed` → 立刻发红色告警。同一异常只报一次，恢复后发绿色恢复卡。
2. **正常时定期进度播报（蓝卡）**：进度百分比+进度条、当前阶段、已运行多久、预计剩余、速度、loss。
3. 训练完成 → 绿色完成卡。
4. 全表快照写到 `status.md`，也可直接看这个文档。

播报标题统一为：**`watchdog播报：任务 <job_id>（<名字>）`**。

## 支持多种任务类型（可扩展）

用统一的 `JobReport`（字段全部可选）承载不同任务，谁能填就填：

| 任务类型 | 状态 | 进度 | 速度/ETA | CPU/GPU/内存 | 已运行 |
|---|---|---|---|---|---|
| **训练**（有 TB） | TB 新鲜度推断 | step/max_steps | ✅ | GPU=world_size；CPU/内存需 ct | ✅ |
| **通用**（给日志路径） | 日志新鲜度推断 | 日志里的 X/Y（可选） | 部分 | 需 ct | ✅ |
| **未知**（啥都没给） | 需 ct | — | — | 需 ct | — |

> ⚠️ 本机目前**没有** `ct`/平台客户端，所以 **CPU/内存、以及权威状态（Running/Failed/Killed）暂时拿不到**；
> 这些字段来自线上平台。代码已预留 `ct` 接入点：本机一旦有 `ct`，`--use-ct`（或 config 里开）就能对**任意任务**自动点亮 CPU/GPU/状态。
> 非训练任务在提供日志路径后，可用「日志新鲜度 + X/Y 进度」监控。

## 快速开始

### 0. 配飞书 webhook（收播报的前提）

编辑 `config.json`，把 `feishu_webhook` 填上你的机器人地址，然后：

```bash
cd /user/hongchenye/watchdog
source /root/minicpmo/bin/activate
python watchdog.py test-notify        # 发一条测试消息确认能收到
```

### 1. 增删查改（只认 job id）

```bash
python watchdog.py add 141369 stage2训练     # 加入监控表（自动识别是不是训练任务）
python watchdog.py add 141369                # 不给名字则默认用「启动时间」当名字
python watchdog.py ls                        # 查看监控表 + 各 job 状态/进度
python watchdog.py show 141369               # 现算并打印某 job 详细快照
python watchdog.py rm 141369                 # 从监控表移除

# 非训练任务：给它日志路径，就能按日志新鲜度 + X/Y 进度监控
python watchdog.py add 141394 数据生成 --log /path/to/xxx.log
```

### 2. 开 / 关

```bash
python watchdog.py start        # 开启守护进程（读 config.json 里的间隔）
python watchdog.py stop         # 关闭
python watchdog.py restart      # 改完参数后重启
```

### 3. 看状态

```bash
python watchdog.py ls           # 表格总览
cat status.md                   # 全表快照文档
tail -f watchdog.log            # 运行日志
```

## 参数（在 config.json 里改，或 start 时用 `--` 覆盖）

| 键 | 默认 | 说明 |
|---|---|---|
| `interval` | 120 | 状态检查间隔（秒） |
| `report_interval` | 600 | 正常时进度播报间隔（秒） |
| `stall` | 1200 | 多久无更新判定卡死（秒） |
| `startup_grace` | 3600 | 启动宽限期（秒），期间无数据不算异常 |
| `search_roots` | 见下 | 按 job id 找 `all_config.json` 的搜索根目录 |
| `feishu_webhook` | "" | 飞书机器人 webhook |
| `use_ct` | false | 本机有 `ct` 时开启，拿权威状态/资源 |

`search_roots` 默认：`train_output/ckpt`、`train_output`、`/backup/.../train/ckpt`。

> 测试期可把 `interval`/`report_interval` 调成 30，观察效果满意后改回 `report_interval: 600`（10 分钟）。

## 目录结构

```
watchdog/
  watchdog.py       # 主程序：守护进程 + 监控表 CLI + 统一 JobReport
  tb_reader.py      # TensorBoard event 增量解析
  notifier.py       # 飞书卡片通知
  config.json       # 配置（gitignore）
  watchlist.txt     # 监控表：<job_id> <名字>
  jobs_extra.json   # 每 job 高级配置：日志路径 / 手动 config（gitignore）
  state/<id>.json   # 每 job 的通知/进度状态（gitignore）
  status.md         # 全表快照
  watchdog.log  events.jsonl  watchdog.pid
```
