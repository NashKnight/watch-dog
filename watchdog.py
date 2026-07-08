#!/usr/bin/env python3
"""训练任务 watchdog: 监控 Cybertron 训练 job 的状态与进度, 自动发飞书。

核心能力 (对应需求):
  1. 状态异常 (卡死无进度 / Failed / Killed) -> 立即发红色告警卡片。
  2. 状态正常时, 每隔一段时间汇报进度: 百分比 / 当前阶段 / 已运行时长 / 预计剩余。
  3. 同时把快照写到 session 目录的 status.md, 方便直接看文档。

数据来源 (登录/notebook 节点即可访问的共享盘):
  - TensorBoard event 文件 (增量解析): 当前 step / loss / wall_time -> 进度、速度、ETA、是否卡死。
  - checkpoint 目录: 已存出的 ckpt (确认进度里程碑)。
  - 可选 `ct job get <id>`: 若本机有 ct, 用它拿权威状态 (Failed/Killed)。

推荐用法:
  python watchdog.py start --config <exp_ckpt_dir>/all_config.json --name my_run
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from tb_reader import (
    TBTailer,
    find_latest_event_file,
    parse_job_id_from_path,
    parse_start_ts_from_path,
)
from notifier import FeishuNotifier, load_webhook_from_config


ROOT = Path(__file__).resolve().parent
SESSIONS_DIR = ROOT / "sessions"
DEFAULT_CONFIG = str(ROOT / "config.json")


# ----------------------------- 工具函数 -----------------------------

def now_ts() -> float:
    return time.time()


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", text).strip("_")[:120] or "run"


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0 or math.isinf(seconds) or math.isnan(seconds):
        return "未知"
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if d:
        parts.append(f"{d}天")
    if h:
        parts.append(f"{h}小时")
    if m:
        parts.append(f"{m}分")
    if not parts:
        parts.append(f"{s}秒")
    return "".join(parts)


def progress_bar(pct: float, width: int = 20) -> str:
    pct = max(0.0, min(1.0, pct))
    filled = int(round(pct * width))
    return "█" * filled + "░" * (width - filled)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, data: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


# ----------------------------- 运行配置 -----------------------------

@dataclass
class RunConfig:
    name: str
    tb_dir: str
    max_steps: int
    ckpt_dir: Optional[str] = None
    warmup_t: int = 0
    t_initial: int = 0
    save_step: int = 0
    prefix: str = ""
    world_size: int = 0
    epochs: int = 0
    job_id: Optional[str] = None
    start_ts: Optional[float] = None


def _read_all_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_run_config(args: argparse.Namespace) -> RunConfig:
    """从 --config all_config.json 或显式参数构造 RunConfig。"""
    tb_dir = args.tb_dir
    max_steps = args.max_steps
    ckpt_dir = args.ckpt_dir
    warmup_t = t_initial = save_step = world_size = epochs = 0
    prefix = ""

    config_path = args.config
    # 允许 --ckpt-dir 指向 exp_ckpt_dir, 自动找里面的 all_config.json
    if not config_path and ckpt_dir:
        cand = os.path.join(ckpt_dir, "all_config.json")
        if os.path.exists(cand):
            config_path = cand

    if config_path and os.path.exists(config_path):
        cfg = _read_all_config(config_path)
        ta = cfg.get("training_args", cfg)
        tb_dir = tb_dir or ta.get("tensorboard")
        max_steps = max_steps or ta.get("max_steps")
        ckpt_dir = ckpt_dir or ta.get("exp_ckpt_dir")
        warmup_t = ta.get("warmup_t", 0)
        t_initial = ta.get("t_initial", 0)
        save_step = ta.get("save_step", 0)
        prefix = ta.get("prefix", "")
        world_size = ta.get("world_size", 0)
        epochs = ta.get("epochs", 0)

    if not tb_dir:
        raise SystemExit("错误: 无法确定 TensorBoard 目录, 请用 --config 或 --tb-dir 指定。")
    if not max_steps:
        raise SystemExit("错误: 无法确定 max_steps, 请用 --config 或 --max-steps 指定。")

    job_id = args.job_id or parse_job_id_from_path(tb_dir)
    start_ts = parse_start_ts_from_path(tb_dir)
    name = args.name or (f"job_{job_id}" if job_id else safe_name(prefix or "run"))

    return RunConfig(
        name=name,
        tb_dir=tb_dir,
        max_steps=int(max_steps),
        ckpt_dir=ckpt_dir,
        warmup_t=int(warmup_t or 0),
        t_initial=int(t_initial or 0),
        save_step=int(save_step or 0),
        prefix=prefix,
        world_size=int(world_size or 0),
        epochs=int(epochs or 0),
        job_id=job_id,
        start_ts=start_ts,
    )


# ----------------------------- 指标 & 状态 -----------------------------

@dataclass
class Metrics:
    have_data: bool
    step: int
    max_steps: int
    pct: float
    steps_per_sec: Optional[float]
    sec_per_step: Optional[float]
    elapsed_sec: Optional[float]
    eta_sec: Optional[float]
    finish_at: Optional[str]
    phase: str
    next_save_step: Optional[int]
    losses: dict
    freshness_sec: Optional[float]
    latest_ckpt_step: Optional[int]


def latest_ckpt_step(rc: RunConfig) -> Optional[int]:
    if not rc.ckpt_dir or not os.path.isdir(rc.ckpt_dir):
        return None
    steps = []
    for p in glob.glob(os.path.join(rc.ckpt_dir, "job_*_ckpt_*")):
        m = re.search(r"_ckpt_(\d+)$", p)
        if m:
            steps.append(int(m.group(1)))
    for p in glob.glob(os.path.join(rc.ckpt_dir, "global_step*")):
        m = re.search(r"global_step(\d+)$", p)
        if m:
            steps.append(int(m.group(1)))
    return max(steps) if steps else None


def build_metrics(rc: RunConfig, tailer: TBTailer, now: float) -> Metrics:
    have_data = tailer.latest_step >= 0
    step = max(tailer.latest_step, 0)
    pct = (step / rc.max_steps) if rc.max_steps else 0.0

    sps = tailer.steps_per_sec()
    spstep = (1.0 / sps) if sps else None
    eta_sec = ((rc.max_steps - step) / sps) if (sps and step < rc.max_steps) else (0.0 if step >= rc.max_steps else None)
    finish_at = None
    if eta_sec is not None:
        finish_at = (datetime.now() + timedelta(seconds=eta_sec)).strftime("%m-%d %H:%M")

    elapsed = (now - rc.start_ts) if rc.start_ts else None

    if rc.warmup_t and step <= rc.warmup_t:
        phase = f"warmup 预热 ({step}/{rc.warmup_t})"
    else:
        phase = "正式训练"

    next_save = None
    if rc.save_step and step < rc.max_steps:
        next_save = ((step // rc.save_step) + 1) * rc.save_step
        next_save = min(next_save, rc.max_steps)

    mtime = tailer.event_file_mtime()
    fresh = (now - mtime) if mtime else (now - tailer.latest_wall if tailer.latest_wall else None)

    return Metrics(
        have_data=have_data,
        step=step,
        max_steps=rc.max_steps,
        pct=pct,
        steps_per_sec=sps,
        sec_per_step=spstep,
        elapsed_sec=elapsed,
        eta_sec=eta_sec,
        finish_at=finish_at,
        phase=phase,
        next_save_step=next_save,
        losses=dict(tailer.losses),
        freshness_sec=fresh,
        latest_ckpt_step=latest_ckpt_step(rc),
    )


# 状态常量
S_WARMUP = "WarmingUp"     # 还没有任何 TB 数据 (进程刚起/加载中)
S_RUNNING = "Running"
S_STALLED = "Stalled"       # 有数据但长时间无进度 -> 疑似卡死/被杀
S_FAILED = "Failed"         # ct 报告 Failed
S_KILLED = "Killed"         # ct 报告 Killed
S_COMPLETED = "Completed"

ABNORMAL = {S_STALLED, S_FAILED, S_KILLED}


def ct_status(job_id: str, ct_cmd: str, workspace: Optional[str]) -> Optional[str]:
    """尽力用 ct 拿权威状态; ct 不可用时返回 None (不报错)。"""
    exe = shutil.which(ct_cmd) or (ct_cmd if os.path.exists(ct_cmd) else None)
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [exe, "--fields", "id,status", "job", "get", str(job_id)],
            cwd=workspace or None,
            text=True,
            capture_output=True,
            timeout=60,
        )
        if proc.returncode != 0:
            return None
        data = json.loads(proc.stdout).get("data", {})
        return data.get("status")
    except Exception:
        return None


def determine_status(
    metrics: Metrics,
    rc: RunConfig,
    now: float,
    stall_sec: int,
    startup_grace_sec: int,
    ct_state: Optional[str],
) -> str:
    # ct 权威状态优先 (若可用)
    if ct_state in ("Failed",):
        return S_FAILED
    if ct_state in ("Killed",):
        return S_KILLED
    if ct_state in ("Succeeded",) or metrics.step >= rc.max_steps and metrics.have_data:
        return S_COMPLETED

    if not metrics.have_data:
        # 还没有 TB 数据: 启动宽限期内算预热, 超过且文件也不更新则疑似卡死
        age = (now - rc.start_ts) if rc.start_ts else 0
        if age > startup_grace_sec and (metrics.freshness_sec or 0) > stall_sec:
            return S_STALLED
        return S_WARMUP

    if (metrics.freshness_sec or 0) > stall_sec:
        return S_STALLED
    return S_RUNNING


# ----------------------------- 消息渲染 -----------------------------

def render_card(rc: RunConfig, m: Metrics, status: str, extra: str = "") -> str:
    lines = []
    head = {
        S_WARMUP: "任务启动中 (加载/编译, 暂无进度数据)",
        S_RUNNING: "训练进行中",
        S_STALLED: "长时间无进度更新, 疑似卡死/被杀",
        S_FAILED: "任务 Failed",
        S_KILLED: "任务 Killed",
        S_COMPLETED: "训练已完成",
    }.get(status, status)
    lines.append(f"**状态**: {status} — {head}")
    ident = rc.prefix or rc.name
    lines.append(f"**实验**: {ident}")
    if rc.job_id:
        lines.append(f"**Job**: {rc.job_id}" + (f"  |  **卡数**: {rc.world_size}" if rc.world_size else ""))

    if m.have_data:
        lines.append(
            f"**进度**: {m.step}/{m.max_steps}  ({m.pct*100:.1f}%)\n`{progress_bar(m.pct)}`"
        )
        lines.append(f"**阶段**: {m.phase}")
        if m.next_save_step:
            lines.append(f"**下次存 ckpt**: step {m.next_save_step}"
                         + (f" (已存到 {m.latest_ckpt_step})" if m.latest_ckpt_step else ""))
        if m.sec_per_step:
            sph = 3600.0 / m.sec_per_step
            lines.append(f"**速度**: {m.sec_per_step:.1f} 秒/step  (~{sph:.0f} step/小时)")
        lines.append(f"**已运行**: {fmt_duration(m.elapsed_sec)}")
        if status not in (S_COMPLETED,):
            lines.append(f"**预计剩余**: {fmt_duration(m.eta_sec)}"
                         + (f"  (约 {m.finish_at} 完成)" if m.finish_at else ""))
        if m.losses:
            loss_str = "  ".join(
                f"{k.split('/')[-1]}={v:.3f}" for k, v in m.losses.items()
            )
            lines.append(f"**Loss**: {loss_str}")
    else:
        lines.append(f"**已运行**: {fmt_duration(m.elapsed_sec)} (尚未产出训练步)")

    if m.freshness_sec is not None:
        lines.append(f"**TB 最近更新**: {fmt_duration(m.freshness_sec)}前")
    lines.append(f"**检查时间**: {now_str()}")
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def level_for(status: str) -> str:
    if status in (S_FAILED, S_KILLED, S_STALLED):
        return "alert"
    if status == S_COMPLETED:
        return "ok"
    return "info"


def write_status_md(path: Path, rc: RunConfig, m: Metrics, status: str) -> None:
    content = "# Watchdog 状态快照\n\n" + render_card(rc, m, status).replace("**", "**") + "\n"
    path.write_text(content, encoding="utf-8")


# ----------------------------- 监控主循环 -----------------------------

def monitor(args: argparse.Namespace) -> None:
    rc = build_run_config(args)
    session_dir = Path(args.session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "pid").write_text(str(os.getpid()), encoding="utf-8")

    log_path = session_dir / "watchdog.log"
    state_path = session_dir / "state.json"
    events_path = session_dir / "events.jsonl"
    status_md = session_dir / "status.md"

    def log(msg: str) -> None:
        line = f"[{now_str()}] {msg}"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    webhook = args.webhook or load_webhook_from_config(args.config_file) or None
    notifier = FeishuNotifier(webhook=webhook, logger=log)

    tailer = TBTailer(tb_dir=rc.tb_dir)

    state = load_json(state_path, {})
    state.update({
        "name": rc.name,
        "job_id": rc.job_id,
        "prefix": rc.prefix,
        "tb_dir": rc.tb_dir,
        "ckpt_dir": rc.ckpt_dir,
        "max_steps": rc.max_steps,
        "created_at": state.get("created_at", now_str()),
    })
    started_notified = state.get("started_notified", False)
    last_report_ts = state.get("last_report_ts", 0)
    incident = state.get("incident")  # 当前未恢复的异常状态
    completed_notified = state.get("completed_notified", False)
    save_json(state_path, state)

    log(f"watchdog 启动 pid={os.getpid()} job={rc.job_id} tb_dir={rc.tb_dir}")
    log(f"参数: interval={args.interval}s report={args.report_interval}s "
        f"stall={args.stall}s startup_grace={args.startup_grace}s notify={'on' if notifier.enabled else 'off'}")

    while True:
        now = now_ts()
        try:
            tailer.poll()
        except Exception as e:  # noqa: BLE001
            log(f"读取 TB 失败: {e}")

        ct_state = None
        if args.use_ct and rc.job_id:
            ct_state = ct_status(rc.job_id, args.ct_cmd, args.workspace)

        m = build_metrics(rc, tailer, now)
        status = determine_status(m, rc, now, args.stall, args.startup_grace, ct_state)

        # 落盘
        state.update({
            "last_status": status,
            "last_checked_at": now_str(),
            "step": m.step,
            "pct": round(m.pct, 4),
            "sec_per_step": m.sec_per_step,
            "eta_sec": m.eta_sec,
            "freshness_sec": m.freshness_sec,
            "losses": m.losses,
            "ct_state": ct_state,
        })
        append_jsonl(events_path, {
            "time": now_str(), "status": status, "step": m.step,
            "pct": round(m.pct, 4), "freshness_sec": m.freshness_sec, "ct_state": ct_state,
        })
        try:
            write_status_md(status_md, rc, m, status)
        except Exception as e:  # noqa: BLE001
            log(f"写 status.md 失败: {e}")
        log(f"status={status} step={m.step}/{m.max_steps} pct={m.pct*100:.1f}% "
            f"fresh={fmt_duration(m.freshness_sec)} ct={ct_state}")

        # ---- 通知状态机 ----
        if not started_notified:
            notifier.send(f"🚀 Watchdog 已启动 · {rc.prefix or rc.name}",
                          render_card(rc, m, status), level="info")
            started_notified = True
            last_report_ts = now

        # 完成
        if status == S_COMPLETED:
            if not completed_notified:
                notifier.send(f"✅ 训练完成 · {rc.prefix or rc.name}",
                              render_card(rc, m, S_COMPLETED), level="ok")
                completed_notified = True
                log("训练完成。")
            state.update({"started_notified": started_notified, "last_report_ts": last_report_ts,
                          "incident": None, "completed_notified": completed_notified})
            save_json(state_path, state)
            if not args.keep_alive:
                return
            time.sleep(args.interval)
            continue

        # 异常: 进入新异常时告警 (同一异常不重复刷屏)
        if status in ABNORMAL:
            if incident != status:
                extra = ""
                if status == S_STALLED:
                    extra = ("> ⚠️ TB 日志长时间未更新, 训练可能已卡死或被杀。"
                             "请人工确认 (查看 job 状态 / 节点)。")
                notifier.send(f"🚨 任务异常告警 · {rc.prefix or rc.name}",
                              render_card(rc, m, status, extra), level="alert")
                incident = status
                log(f"触发告警: {status}")
            # Failed/Killed 属终态, 除非 keep-alive 否则退出
            if status in (S_FAILED, S_KILLED) and not args.keep_alive:
                state.update({"started_notified": started_notified, "last_report_ts": last_report_ts,
                              "incident": incident, "completed_notified": completed_notified})
                save_json(state_path, state)
                return
        else:
            # 恢复: 之前有异常, 现在正常了
            if incident is not None:
                notifier.send(f"🟢 已恢复正常 · {rc.prefix or rc.name}",
                              render_card(rc, m, status,
                                          "> 训练已重新产生进度, 从之前的异常状态恢复。"),
                              level="ok")
                log(f"从 {incident} 恢复到 {status}")
                incident = None
                last_report_ts = now

            # 定期进度汇报
            if status == S_RUNNING and (now - last_report_ts) >= args.report_interval:
                notifier.send(f"📊 训练进度 · {rc.prefix or rc.name}",
                              render_card(rc, m, status), level="info")
                last_report_ts = now
                log("发送定期进度汇报。")

        state.update({"started_notified": started_notified, "last_report_ts": last_report_ts,
                      "incident": incident, "completed_notified": completed_notified})
        save_json(state_path, state)
        time.sleep(args.interval)


# ----------------------------- CLI 子命令 -----------------------------

def _add_common_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="all_config.json 路径 (自动读取 tb_dir/max_steps 等)")
    p.add_argument("--tb-dir", help="TensorBoard 目录 (不用 --config 时指定)")
    p.add_argument("--ckpt-dir", help="exp_ckpt_dir (可自动定位 all_config.json)")
    p.add_argument("--max-steps", type=int, help="总步数 (不用 --config 时指定)")
    p.add_argument("--job-id", help="Cybertron job id (缺省从 tb 路径解析)")
    p.add_argument("--name", help="session 名 (缺省 job_<id>)")
    p.add_argument("--interval", type=int, default=120, help="状态检查间隔秒, 默认120")
    p.add_argument("--report-interval", type=int, default=3600, help="正常时进度汇报间隔秒, 默认3600")
    p.add_argument("--stall", type=int, default=1200, help="多久无更新判定卡死(秒), 默认1200")
    p.add_argument("--startup-grace", type=int, default=3600, help="启动宽限期(秒), 期间无数据不算异常, 默认3600")
    p.add_argument("--webhook", help="飞书 webhook (优先级最高)")
    p.add_argument("--config-file", default=DEFAULT_CONFIG, help="含 feishu_webhook 的配置文件")
    p.add_argument("--use-ct", action="store_true", help="尝试用 ct 拿权威状态 (本机需有 ct)")
    p.add_argument("--ct-cmd", default="ct", help="ct 可执行名/路径")
    p.add_argument("--workspace", default=None, help="ct 执行的工作目录")
    p.add_argument("--keep-alive", action="store_true", help="终态后不退出, 继续记录")


def cmd_start(args: argparse.Namespace) -> None:
    rc = build_run_config(args)  # 提前校验配置
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    session_name = safe_name(args.name or rc.name)
    session_dir = SESSIONS_DIR / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    # 转发参数给后台 run
    cmd = [sys.executable, str(Path(__file__).resolve()), "run",
           "--session-dir", str(session_dir),
           "--interval", str(args.interval),
           "--report-interval", str(args.report_interval),
           "--stall", str(args.stall),
           "--startup-grace", str(args.startup_grace),
           "--ct-cmd", args.ct_cmd,
           "--config-file", args.config_file]
    if args.config:
        cmd += ["--config", args.config]
    if args.tb_dir:
        cmd += ["--tb-dir", args.tb_dir]
    if args.ckpt_dir:
        cmd += ["--ckpt-dir", args.ckpt_dir]
    if args.max_steps:
        cmd += ["--max-steps", str(args.max_steps)]
    if args.job_id:
        cmd += ["--job-id", args.job_id]
    cmd += ["--name", session_name]
    if args.webhook:
        cmd += ["--webhook", args.webhook]
    if args.use_ct:
        cmd += ["--use-ct"]
    if args.workspace:
        cmd += ["--workspace", args.workspace]
    if args.keep_alive:
        cmd += ["--keep-alive"]

    if args.foreground:
        args.session_dir = str(session_dir)
        monitor(args)
        return

    out = (session_dir / "watchdog.stdout").open("a", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT,
                            start_new_session=True)
    (session_dir / "pid").write_text(str(proc.pid), encoding="utf-8")
    print(f"已后台启动 watchdog: pid={proc.pid}")
    print(f"session 目录: {session_dir}")
    print(f"实时状态文档: {session_dir / 'status.md'}")
    print(f"日志: {session_dir / 'watchdog.log'}")


def cmd_status(args: argparse.Namespace) -> None:
    if not SESSIONS_DIR.exists():
        print("暂无 session。")
        return
    for session in sorted(SESSIONS_DIR.glob("*")):
        if not session.is_dir():
            continue
        st = load_json(session / "state.json", {})
        pid_file = session / "pid"
        pid = pid_file.read_text().strip() if pid_file.exists() else "?"
        alive = "?"
        if pid.isdigit():
            alive = "alive" if os.path.exists(f"/proc/{pid}") else "dead"
        step = st.get("step")
        pct = st.get("pct")
        pct_s = f"{pct*100:.1f}%" if isinstance(pct, (int, float)) else "-"
        print(f"{session.name}\tpid={pid}({alive})\tstatus={st.get('last_status')}\t"
              f"step={step}\tpct={pct_s}\tlast={st.get('last_checked_at')}")


def cmd_stop(args: argparse.Namespace) -> None:
    session = SESSIONS_DIR / safe_name(args.name)
    pid_file = session / "pid"
    if not pid_file.exists():
        print(f"未找到 session: {args.name}")
        return
    pid = pid_file.read_text().strip()
    if pid.isdigit():
        try:
            os.kill(int(pid), 15)
            print(f"已发送 SIGTERM 给 pid={pid}")
        except ProcessLookupError:
            print(f"进程 {pid} 已不存在。")
    else:
        print("pid 无效。")


def cmd_once(args: argparse.Namespace) -> None:
    """单次检查并打印报告 (可选发一条飞书), 用于调试。"""
    rc = build_run_config(args)
    tailer = TBTailer(tb_dir=rc.tb_dir)
    tailer.poll()
    now = now_ts()
    ct_state = ct_status(rc.job_id, args.ct_cmd, args.workspace) if (args.use_ct and rc.job_id) else None
    m = build_metrics(rc, tailer, now)
    status = determine_status(m, rc, now, args.stall, args.startup_grace, ct_state)
    print(render_card(rc, m, status))
    if args.send:
        webhook = args.webhook or load_webhook_from_config(args.config_file)
        FeishuNotifier(webhook=webhook).send(
            f"📊 训练进度(手动) · {rc.prefix or rc.name}", render_card(rc, m, status),
            level=level_for(status))


def cmd_test_notify(args: argparse.Namespace) -> None:
    webhook = args.webhook or load_webhook_from_config(args.config_file)
    n = FeishuNotifier(webhook=webhook)
    if not n.enabled:
        print("未配置 webhook。请用 --webhook 或在 config.json 里填 feishu_webhook。")
        return
    ok = n.send("✅ Watchdog 测试消息",
                f"这是一条来自 watchdog 的测试消息。\n**时间**: {now_str()}", level="ok")
    print("发送成功。" if ok else "发送失败, 请检查 webhook。")


def main() -> int:
    parser = argparse.ArgumentParser(description="训练任务 watchdog (状态告警 + 进度汇报 + 飞书通知)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser("start", help="后台启动一个 watchdog")
    _add_common_run_args(p_start)
    p_start.add_argument("--foreground", action="store_true", help="前台运行(调试)")
    p_start.set_defaults(func=cmd_start)

    p_run = sub.add_parser("run", help="(内部)前台运行监控循环")
    _add_common_run_args(p_run)
    p_run.add_argument("--session-dir", required=True)
    p_run.set_defaults(func=monitor)

    p_once = sub.add_parser("once", help="单次检查并打印报告")
    _add_common_run_args(p_once)
    p_once.add_argument("--send", action="store_true", help="同时发一条飞书")
    p_once.set_defaults(func=cmd_once)

    p_status = sub.add_parser("status", help="列出所有 session")
    p_status.set_defaults(func=cmd_status)

    p_stop = sub.add_parser("stop", help="停止某个 session")
    p_stop.add_argument("name")
    p_stop.set_defaults(func=cmd_stop)

    p_test = sub.add_parser("test-notify", help="发送一条飞书测试消息")
    p_test.add_argument("--webhook")
    p_test.add_argument("--config-file", default=DEFAULT_CONFIG)
    p_test.set_defaults(func=cmd_test_notify)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
