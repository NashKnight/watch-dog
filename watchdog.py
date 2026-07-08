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
REGISTRY = ROOT / "registry.json"

# 按 job id 自动发现 all_config.json 时搜索的根目录 (可在 config.json 里用 search_roots 覆盖)
DEFAULT_SEARCH_ROOTS = [
    "/user/hongchenye/train_output/ckpt",
    "/user/hongchenye/train_output",
    "/backup/user/hongchenye/train/ckpt",
]

# 监控参数默认值 (add/set 未显式指定时使用)
OPTION_DEFAULTS = {
    "interval": 120,
    "report_interval": 3600,
    "stall": 1200,
    "startup_grace": 3600,
    "use_ct": False,
    "ct_cmd": "ct",
    "workspace": None,
    "keep_alive": False,
    "webhook": None,
}


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


# ----------------------------- 自动发现 & 注册表 -----------------------------

def get_search_roots(config_file: str = DEFAULT_CONFIG) -> list:
    if config_file and os.path.exists(config_file):
        try:
            roots = json.load(open(config_file, encoding="utf-8")).get("search_roots")
            if roots:
                return list(roots)
        except Exception:
            pass
    return list(DEFAULT_SEARCH_ROOTS)


def _iter_config_paths(roots: list):
    """在若干根目录下按有限深度查找 all_config.json (避免深挖 ckpt 分片目录)。"""
    seen = set()
    for r in roots:
        if not os.path.isdir(r):
            continue
        for depth in range(1, 7):
            pat = os.path.join(r, *(["*"] * depth), "all_config.json")
            for p in glob.glob(pat):
                if p not in seen:
                    seen.add(p)
                    yield p


def discover_config_by_job_id(job_id: str, roots: list) -> Optional[str]:
    """根据 job id 找到对应的 all_config.json; 多个匹配取启动时间最新的。"""
    cands = []
    for p in _iter_config_paths(roots):
        try:
            ta = json.load(open(p, encoding="utf-8")).get("training_args", {})
        except Exception:
            continue
        tb = ta.get("tensorboard") or ""
        if f"job_{job_id}" in tb or f"job_{job_id}" in p:
            start = parse_start_ts_from_path(tb) or os.path.getmtime(p)
            cands.append((start, p))
    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0][1]


def load_registry() -> dict:
    return load_json(REGISTRY, {})


def save_registry(d: dict) -> None:
    save_json(REGISTRY, d)


def session_dir_for(name: str) -> Path:
    return SESSIONS_DIR / safe_name(name)


def read_pid(session_dir: Path) -> Optional[str]:
    p = session_dir / "pid"
    return p.read_text().strip() if p.exists() else None


def is_alive(session_dir: Path) -> bool:
    pid = read_pid(session_dir)
    return bool(pid and pid.isdigit() and os.path.exists(f"/proc/{pid}"))


def stop_session(session_dir: Path) -> bool:
    pid = read_pid(session_dir)
    if pid and pid.isdigit():
        try:
            os.kill(int(pid), 15)
            return True
        except ProcessLookupError:
            return False
    return False


# ----------------------------- 启动后台监控 -----------------------------

def spawn_watchdog(name: str, launch: dict) -> int:
    """根据 launch 配置字典后台拉起一个 run 进程, 返回 pid。"""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    session_dir = session_dir_for(name)
    session_dir.mkdir(parents=True, exist_ok=True)

    opts = {**OPTION_DEFAULTS, **launch.get("options", {})}
    cmd = [sys.executable, str(Path(__file__).resolve()), "run",
           "--session-dir", str(session_dir),
           "--name", safe_name(name),
           "--interval", str(opts["interval"]),
           "--report-interval", str(opts["report_interval"]),
           "--stall", str(opts["stall"]),
           "--startup-grace", str(opts["startup_grace"]),
           "--ct-cmd", str(opts["ct_cmd"]),
           "--config-file", launch.get("config_file", DEFAULT_CONFIG)]
    if launch.get("config"):
        cmd += ["--config", launch["config"]]
    if launch.get("tb_dir"):
        cmd += ["--tb-dir", launch["tb_dir"]]
    if launch.get("ckpt_dir"):
        cmd += ["--ckpt-dir", launch["ckpt_dir"]]
    if launch.get("max_steps"):
        cmd += ["--max-steps", str(launch["max_steps"])]
    if launch.get("job_id"):
        cmd += ["--job-id", str(launch["job_id"])]
    if opts.get("webhook"):
        cmd += ["--webhook", opts["webhook"]]
    if opts.get("use_ct"):
        cmd += ["--use-ct"]
    if opts.get("workspace"):
        cmd += ["--workspace", opts["workspace"]]
    if opts.get("keep_alive"):
        cmd += ["--keep-alive"]

    out = (session_dir / "watchdog.stdout").open("a", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT,
                            start_new_session=True)
    (session_dir / "pid").write_text(str(proc.pid), encoding="utf-8")
    return proc.pid


def _ns_for_build(config=None, tb_dir=None, ckpt_dir=None, max_steps=None,
                  job_id=None, name=None) -> argparse.Namespace:
    return argparse.Namespace(config=config, tb_dir=tb_dir, ckpt_dir=ckpt_dir,
                              max_steps=max_steps, job_id=job_id, name=name)


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


def _options_from_args(args: argparse.Namespace) -> dict:
    """把命令行里显式给出的监控参数收集成 options (只收非默认/显式项)。"""
    opts = {}
    for key in ("interval", "report_interval", "stall", "startup_grace",
                "use_ct", "ct_cmd", "workspace", "keep_alive", "webhook"):
        val = getattr(args, key, None)
        if val is not None:
            opts[key] = val
    return opts


def cmd_add(args: argparse.Namespace) -> None:
    """核心命令: watchdog add <job_id> —— 自动发现配置并启动监控, 登记到监控表。"""
    reg = load_registry()
    roots = get_search_roots(args.config_file)
    cli_opts = _options_from_args(args)

    for job_id in args.job_ids:
        job_id = str(job_id)
        config = args.config
        tb_dir = args.tb_dir
        max_steps = args.max_steps
        ckpt_dir = args.ckpt_dir

        if not config and not tb_dir:
            config = discover_config_by_job_id(job_id, roots)
            if not config:
                print(f"[{job_id}] 未能自动定位 all_config.json。")
                print("  可手动指定: --config <all_config.json> 或 --tb-dir <dir> --max-steps <N>")
                print(f"  (搜索根目录: {roots})")
                continue
            print(f"[{job_id}] 已自动定位配置: {config}")

        # 校验并抽取 tb_dir / max_steps / name
        try:
            rc = build_run_config(_ns_for_build(config=config, tb_dir=tb_dir,
                                                ckpt_dir=ckpt_dir, max_steps=max_steps,
                                                job_id=job_id, name=args.name))
        except SystemExit as e:
            print(f"[{job_id}] 配置无效: {e}")
            continue

        name = safe_name(args.name or rc.name or f"job_{job_id}")
        sdir = session_dir_for(name)
        if name in reg and is_alive(sdir):
            print(f"[{job_id}] 已在监控中 (session={name}, pid={read_pid(sdir)})。"
                  f" 如需改参数用: watchdog set {job_id} ...")
            continue

        options = {**OPTION_DEFAULTS, **cli_opts}
        launch = {
            "job_id": job_id,
            "config": config,
            "tb_dir": rc.tb_dir,
            "ckpt_dir": rc.ckpt_dir,
            "max_steps": rc.max_steps,
            "options": options,
            "config_file": args.config_file,
        }
        if args.no_start:
            reg[name] = {**launch, "name": name, "session_dir": str(sdir),
                         "added_at": now_str(), "started": False}
            save_registry(reg)
            print(f"[{job_id}] 已登记 (未启动)。用 watchdog restart {job_id} 启动。")
            continue

        pid = spawn_watchdog(name, launch)
        reg[name] = {**launch, "name": name, "session_dir": str(sdir),
                     "added_at": now_str(), "started": True}
        save_registry(reg)
        print(f"[{job_id}] 已启动监控: session={name} pid={pid} "
              f"step_total={rc.max_steps}")
        print(f"         状态文档: {sdir / 'status.md'}")


def _find_entry(reg: dict, key: str):
    """按 job_id 或 session 名找注册项, 返回 (name, entry) 或 (None, None)。"""
    key = str(key)
    if key in reg:
        return key, reg[key]
    for name, e in reg.items():
        if str(e.get("job_id")) == key:
            return name, e
    # 允许用 safe_name 后的名字
    sk = safe_name(key)
    if sk in reg:
        return sk, reg[sk]
    return None, None


def cmd_rm(args: argparse.Namespace) -> None:
    reg = load_registry()
    for key in args.job_ids:
        name, entry = _find_entry(reg, key)
        if not entry:
            print(f"[{key}] 不在监控表中。")
            continue
        sdir = session_dir_for(name)
        if is_alive(sdir):
            stop_session(sdir)
            print(f"[{key}] 已停止 (pid={read_pid(sdir)})。")
        reg.pop(name, None)
        save_registry(reg)
        if args.purge and sdir.exists():
            shutil.rmtree(sdir, ignore_errors=True)
            print(f"[{key}] 已删除 session 目录。")
        print(f"[{key}] 已从监控表移除。")


def cmd_ls(args: argparse.Namespace) -> None:
    reg = load_registry()
    if not reg:
        print("监控表为空。用  watchdog add <job_id>  添加。")
        return
    header = f"{'JOB':<8} {'NAME':<22} {'PID':<8} {'ALIVE':<6} {'STATUS':<10} {'STEP':<12} {'PCT':<7} {'ETA':<12} LAST"
    print(header)
    print("-" * len(header))
    for name, e in reg.items():
        sdir = session_dir_for(name)
        st = load_json(sdir / "state.json", {})
        pid = read_pid(sdir) or "-"
        alive = "yes" if is_alive(sdir) else "no"
        status = st.get("last_status", "-")
        step = st.get("step")
        step_s = f"{step}/{e.get('max_steps','?')}" if step is not None else "-"
        pct = st.get("pct")
        pct_s = f"{pct*100:.1f}%" if isinstance(pct, (int, float)) else "-"
        eta = st.get("eta_sec")
        eta_s = fmt_duration(eta) if isinstance(eta, (int, float)) else "-"
        last = st.get("last_checked_at", "-")
        print(f"{str(e.get('job_id','-')):<8} {name:<22} {pid:<8} {alive:<6} "
              f"{status:<10} {step_s:<12} {pct_s:<7} {eta_s:<12} {last}")


def cmd_show(args: argparse.Namespace) -> None:
    reg = load_registry()
    name, entry = _find_entry(reg, args.job_id)
    if not entry:
        print(f"[{args.job_id}] 不在监控表中。")
        return
    md = session_dir_for(name) / "status.md"
    if md.exists():
        print(md.read_text(encoding="utf-8"))
    else:
        print(f"[{args.job_id}] 尚无 status.md (可能刚启动)。")


def cmd_set(args: argparse.Namespace) -> None:
    """改监控参数并自动重启该 session。"""
    reg = load_registry()
    name, entry = _find_entry(reg, args.job_id)
    if not entry:
        print(f"[{args.job_id}] 不在监控表中。先 add。")
        return
    changed = _options_from_args(args)
    if not changed:
        print("未指定要修改的参数。可改: --interval/--report-interval/--stall/"
              "--startup-grace/--use-ct/--webhook/--keep-alive 等")
        return
    entry.setdefault("options", {}).update(changed)
    reg[name] = entry
    save_registry(reg)
    sdir = session_dir_for(name)
    if is_alive(sdir):
        stop_session(sdir)
        time.sleep(1)
    pid = spawn_watchdog(name, entry)
    entry["started"] = True
    save_registry(reg)
    print(f"[{args.job_id}] 已更新参数 {changed} 并重启 (pid={pid})。")


def cmd_restart(args: argparse.Namespace) -> None:
    reg = load_registry()
    keys = args.job_ids or list(reg.keys())
    for key in keys:
        name, entry = _find_entry(reg, key)
        if not entry:
            print(f"[{key}] 不在监控表中。")
            continue
        sdir = session_dir_for(name)
        if is_alive(sdir):
            stop_session(sdir)
            time.sleep(1)
        pid = spawn_watchdog(name, entry)
        entry["started"] = True
        reg[name] = entry
        save_registry(reg)
        print(f"[{key}] 已重启 (pid={pid})。")


def cmd_stop(args: argparse.Namespace) -> None:
    reg = load_registry()
    keys = args.job_ids or list(reg.keys())
    for key in keys:
        name, entry = _find_entry(reg, key)
        # 也允许直接用 session 名停未登记的
        sdir = session_dir_for(name or key)
        if not sdir.exists():
            print(f"[{key}] 未找到对应 session。")
            continue
        if stop_session(sdir):
            print(f"[{key}] 已停止 (pid={read_pid(sdir)})。")
        else:
            print(f"[{key}] 进程不存在或已停止。")


def cmd_once(args: argparse.Namespace) -> None:
    """单次检查并打印报告 (可选发一条飞书), 用于调试。支持仅给 --job-id 自动发现。"""
    if args.job_id and not args.config and not args.tb_dir:
        cfg = discover_config_by_job_id(str(args.job_id), get_search_roots(args.config_file))
        if cfg:
            args.config = cfg
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


def _add_option_args(p: argparse.ArgumentParser) -> None:
    """add/set 共用的可调监控参数 (默认 None, 表示不覆盖)。"""
    p.add_argument("--interval", type=int, help="状态检查间隔秒 (默认120)")
    p.add_argument("--report-interval", type=int, help="正常时进度汇报间隔秒 (默认3600)")
    p.add_argument("--stall", type=int, help="多久无更新判定卡死(秒) (默认1200)")
    p.add_argument("--startup-grace", type=int, help="启动宽限期(秒) (默认3600)")
    p.add_argument("--webhook", help="飞书 webhook")
    p.add_argument("--use-ct", action="store_true", default=None, help="用 ct 拿权威状态")
    p.add_argument("--ct-cmd", help="ct 可执行名/路径")
    p.add_argument("--workspace", help="ct 执行工作目录")
    p.add_argument("--keep-alive", action="store_true", default=None, help="终态后不退出")
    p.add_argument("--config-file", default=DEFAULT_CONFIG, help="含 feishu_webhook/search_roots 的配置")


def main() -> int:
    parser = argparse.ArgumentParser(description="训练任务 watchdog (监控表: add/rm/ls/set + 飞书告警/进度)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- 监控表核心命令 ----
    p_add = sub.add_parser("add", help="添加 job 到监控表并启动 (只需 job id)")
    p_add.add_argument("job_ids", nargs="+", help="一个或多个 job id")
    p_add.add_argument("--config", help="手动指定 all_config.json (跳过自动发现)")
    p_add.add_argument("--tb-dir", help="手动指定 TB 目录")
    p_add.add_argument("--ckpt-dir", help="手动指定 exp_ckpt_dir")
    p_add.add_argument("--max-steps", type=int, help="手动指定总步数")
    p_add.add_argument("--name", help="自定义 session 名 (缺省 job_<id>)")
    p_add.add_argument("--no-start", action="store_true", help="只登记不启动")
    _add_option_args(p_add)
    p_add.set_defaults(func=cmd_add)

    p_rm = sub.add_parser("rm", help="从监控表移除 (并停止)")
    p_rm.add_argument("job_ids", nargs="+")
    p_rm.add_argument("--purge", action="store_true", help="同时删除 session 目录")
    p_rm.set_defaults(func=cmd_rm)

    p_ls = sub.add_parser("ls", help="查看监控表 (状态/进度)")
    p_ls.set_defaults(func=cmd_ls)

    p_show = sub.add_parser("show", help="查看某 job 的详细状态快照")
    p_show.add_argument("job_id")
    p_show.set_defaults(func=cmd_show)

    p_set = sub.add_parser("set", help="修改某 job 的监控参数并重启")
    p_set.add_argument("job_id")
    _add_option_args(p_set)
    p_set.set_defaults(func=cmd_set)

    p_restart = sub.add_parser("restart", help="重启监控 (缺省全部)")
    p_restart.add_argument("job_ids", nargs="*")
    p_restart.set_defaults(func=cmd_restart)

    p_stop = sub.add_parser("stop", help="停止监控 (缺省全部)")
    p_stop.add_argument("job_ids", nargs="*")
    p_stop.set_defaults(func=cmd_stop)

    p_test = sub.add_parser("test-notify", help="发送一条飞书测试消息")
    p_test.add_argument("--webhook")
    p_test.add_argument("--config-file", default=DEFAULT_CONFIG)
    p_test.set_defaults(func=cmd_test_notify)

    # ---- 底层/调试命令 ----
    p_run = sub.add_parser("run", help="(内部)前台运行监控循环")
    _add_common_run_args(p_run)
    p_run.add_argument("--session-dir", required=True)
    p_run.set_defaults(func=monitor)

    p_once = sub.add_parser("once", help="单次检查并打印报告 (支持仅 --job-id)")
    _add_common_run_args(p_once)
    p_once.add_argument("--send", action="store_true", help="同时发一条飞书")
    p_once.set_defaults(func=cmd_once)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
