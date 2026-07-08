# Train Watchdog · 训练任务看门狗

监控 Cybertron 训练 job 的**状态**与**进度**，并**自动发飞书**，省得自己反复翻日志。

核心功能，正好对应两条需求：

1. **状态异常立即告警**：训练卡死（TB 长时间无更新）/ `Failed` / `Killed` → 立刻发**红色**告警卡片。
2. **正常时定期汇报进度**：每隔一段时间发一张**蓝色**进度卡片，含：
   - 进度百分比 + 进度条（`step / max_steps`）
   - 当前阶段（warmup / 正式训练、下一次存 ckpt 的 step）
   - 已运行多久、预计还要多久、预计完成时间点
   - 训练速度（秒/step）、当前各项 loss
3. 训练完成 → 发**绿色**完成卡片；从异常恢复 → 发绿色恢复卡片。
4. 同时把最新快照写到 `sessions/<name>/status.md`，也可以直接看这个文档。

## 数据从哪来

登录 / notebook 节点就能访问共享盘，因此不依赖登上训练节点：

- **TensorBoard event 文件**（增量解析，记录字节偏移，不全量重读）：当前 step / loss / wall_time → 进度、速度、ETA、是否卡死。
- **checkpoint 目录**：已存出的 ckpt，用于确认里程碑。
- **可选 `ct job get <id>`**：本机若有 `ct`，用它拿权威状态（能明确区分 `Failed` / `Killed`）。没有 `ct` 时，用「TB 长时间不更新」来判定卡死。

> 关键：一个 `all_config.json`（训练会写到 `exp_ckpt_dir/all_config.json`）就包含 `tensorboard` 路径、`max_steps`、`warmup_t`、`save_step` 等，watchdog 可据此**全自动配置**。

## 快速开始

### 0. 配一次飞书 webhook

```bash
cd /user/hongchenye/watchdog
cp config.example.json config.json
# 编辑 config.json，把 feishu_webhook 换成你的机器人地址
python watchdog.py test-notify          # 发一条测试消息确认能收到
```

（也可以用环境变量 `FEISHU_WEBHOOK=...`，或每次 `--webhook ...`。）

### 1. 启动监控（推荐：指向 all_config.json）

```bash
source /root/minicpmo/bin/activate   # 需要有 requests / tensorboard
cd /user/hongchenye/watchdog
python watchdog.py start \
  --config /user/hongchenye/train_output/ckpt/minicpm-v/stage2_omni/0708/stage2_1_omnipro_v1_ablation/all_config.json \
  --name job_141369
```

### 2. 看状态

```bash
python watchdog.py status                    # 列出所有 session
cat sessions/job_141369/status.md            # 最新快照文档
tail -f sessions/job_141369/watchdog.log     # 运行日志
```

### 3. 停止

```bash
python watchdog.py stop job_141369
```

## 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--config` | — | `all_config.json` 路径，自动读 tb_dir / max_steps 等 |
| `--tb-dir` / `--max-steps` | — | 不用 `--config` 时手动指定 |
| `--ckpt-dir` | — | `exp_ckpt_dir`，可自动定位其中的 `all_config.json` |
| `--interval` | 120 | 状态检查间隔（秒） |
| `--report-interval` | 3600 | 正常时进度汇报间隔（秒） |
| `--stall` | 1200 | 多久无更新判定卡死（秒） |
| `--startup-grace` | 3600 | 启动宽限期（秒），期间无数据不算异常 |
| `--use-ct` | 关 | 尝试用 `ct` 拿权威状态（本机需有 `ct`） |
| `--webhook` | — | 飞书 webhook（优先级最高） |
| `--keep-alive` | 关 | 终态后不退出，继续记录 |

## 调试 / 手动看一次

```bash
python watchdog.py once --config <all_config.json>          # 只打印一次报告
python watchdog.py once --config <all_config.json> --send   # 顺便发一条飞书
```

## 目录结构

```
watchdog/
  watchdog.py        # 主程序 (CLI + 监控循环 + 状态机)
  tb_reader.py       # TensorBoard event 增量解析 + 指标
  notifier.py        # 飞书卡片通知
  config.example.json
  sessions/<name>/   # 每个被监控 job 一个目录
    pid  state.json  events.jsonl  watchdog.log  status.md
```

## 说明

- 无 `ct` 时，「异常」主要靠 **TB 长时间无更新** 推断（可能是卡死/被杀）；有 `ct` 时能明确报 `Failed`/`Killed`。
- 同一个异常只告警一次，避免刷屏；恢复后再发一条恢复消息。
- 自动排错 / 自动重提 job（借鉴 `/user/xubokai/watchdog_ctcli` 的 cursor-agent 思路）暂未实现，后续在 `dev_hcy` 分支迭代。
