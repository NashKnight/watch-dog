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

## 快速开始（监控表模式，只用 job id）

### 0. 配一次飞书 webhook

```bash
cd /user/hongchenye/watchdog
cp config.example.json config.json
# 编辑 config.json，把 feishu_webhook 换成你的机器人地址
python watchdog.py test-notify          # 发一条测试消息确认能收到
```

（也可以用环境变量 `FEISHU_WEBHOOK=...`，或每次 `--webhook ...`。）

### 1. 增删查改，全部只认 job id

```bash
source /root/minicpmo/bin/activate      # 需要有 requests / tensorboard
cd /user/hongchenye/watchdog

python watchdog.py add 141369           # 加入监控（自动定位配置并启动）
python watchdog.py add 141369 142000    # 可一次加多个
python watchdog.py ls                   # 查看监控表（状态/进度/ETA 一览）
python watchdog.py show 141369          # 看某个 job 的详细快照
python watchdog.py set 141369 --report-interval 1800 --stall 900   # 改参数（自动重启）
python watchdog.py restart 141369       # 重启（不带 id 则重启全部）
python watchdog.py stop 141369          # 停止（不带 id 则停止全部）
python watchdog.py rm 141369            # 移除（加 --purge 连 session 目录一起删）
```

`add <job_id>` 会自动在 `search_roots` 下按 `job_<id>` 找到对应的 `all_config.json`，
从中读出 TB 目录、`max_steps`、`warmup_t` 等，无需你手输路径。

> 自动发现失败时（例如换了实验根目录），可手动兜底：
> `python watchdog.py add <job_id> --config <all_config.json>`
> 或 `--tb-dir <dir> --max-steps <N>`。
> 搜索根目录可在 `config.json` 里用 `"search_roots": [...]` 覆盖。

### 2. 看状态

```bash
python watchdog.py ls                         # 表格总览
python watchdog.py show 141369                # 详细快照
cat sessions/job_141369/status.md             # 同样是最新快照文档
tail -f sessions/job_141369/watchdog.log      # 运行日志
```

## 命令一览

| 命令 | 说明 |
|---|---|
| `add <job_id...> [选项]` | 加入监控表并启动（自动发现配置）。`--no-start` 只登记不启动 |
| `rm <job_id...> [--purge]` | 停止并移除；`--purge` 连 session 目录删掉 |
| `ls` | 查看监控表：状态 / step / 百分比 / ETA |
| `show <job_id>` | 打印该 job 的详细状态快照 |
| `set <job_id> [选项]` | 修改监控参数并自动重启 |
| `restart [job_id...]` | 重启（缺省全部） |
| `stop [job_id...]` | 停止（缺省全部） |
| `test-notify` | 发一条飞书测试消息 |
| `once [--job-id N / --config ...] [--send]` | 单次检查打印报告（调试） |

## 可调监控参数（add / set 通用）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--interval` | 120 | 状态检查间隔（秒） |
| `--report-interval` | 3600 | 正常时进度汇报间隔（秒） |
| `--stall` | 1200 | 多久无更新判定卡死（秒） |
| `--startup-grace` | 3600 | 启动宽限期（秒），期间无数据不算异常 |
| `--use-ct` | 关 | 尝试用 `ct` 拿权威状态（本机需有 `ct`） |
| `--webhook` | — | 飞书 webhook（优先级最高） |
| `--keep-alive` | 关 | 终态后不退出，继续记录 |

## 目录结构

```
watchdog/
  watchdog.py        # 主程序 (监控表 CLI + 监控循环 + 状态机)
  tb_reader.py       # TensorBoard event 增量解析 + 指标
  notifier.py        # 飞书卡片通知
  config.example.json
  registry.json      # 监控表 (自动维护, gitignore)
  sessions/<name>/   # 每个被监控 job 一个目录
    pid  state.json  events.jsonl  watchdog.log  status.md
```

## 说明

- 无 `ct` 时，「异常」主要靠 **TB 长时间无更新** 推断（可能是卡死/被杀）；有 `ct` 时能明确报 `Failed`/`Killed`。
- 同一个异常只告警一次，避免刷屏；恢复后再发一条恢复消息。
- 自动排错 / 自动重提 job（借鉴 `/user/xubokai/watchdog_ctcli` 的 cursor-agent 思路）暂未实现，后续在 `dev_hcy` 分支迭代。
