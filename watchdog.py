#!/usr/bin/env python3
"""训练/任务 watchdog: 一个常驻守护进程读取"监控表", 监控每个 job 的状态与进度, 自动发飞书。

模型 (符合"一个开关 + 一张监控表"):
  - 监控表 = 纯文本 ``watchlist.txt``, 每行一个 job:  ``<job_id> <任务名>`` (名字可选)。
  - 守护进程 = 一个进程 (``start`` 开 / ``stop`` 关), 每轮重读监控表, 逐个 job 评估并按需播报。
  - 增删查改都只认 job id:  ``add`` / ``rm`` / ``ls`` / ``show``。

数据来源 (登录/notebook 节点即可, 不必登训练节点):
  - 训练任务: TensorBoard event 文件 -> step/loss/进度/速度/ETA;  GPU 卡数取自 all_config.json 的 world_size。
  - 通用任务(带日志路径): 日志文件新鲜度 + 可选 X/Y 进度。
  - 可选 ``ct``: 本机若有 ct, 用它拿任意任务的权威状态 + CPU/GPU/内存/起止时间 (自动点亮)。

设计上用统一的 ``JobReport`` (字段全部可选) 承载不同任务类型, 谁能填就填, 渲染时跳过未知字段 —— 方便扩展。
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
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from tb_reader import (
    TBTailer,
    parse_job_id_from_path,
    parse_start_ts_from_path,
)
from notifier import FeishuNotifier, load_webhook_from_config


ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "config.json"
WATCHLIST = ROOT / "watchlist.txt"
EXTRA_FILE = ROOT / "jobs_extra.json"      # 每个 job 的高级配置 (日志路径/类型覆盖)
STATE_DIR = ROOT / "state"                 # 每个 job 的通知/进度状态
PIDFILE = ROOT / "watchdog.pid"
LOGFILE = ROOT / "watchdog.log"
EVENTS = ROOT / "events.jsonl"
STATUS_MD = ROOT / "status.md"             # 全表快照文档

DEFAULT_SEARCH_ROOTS = [
    "/user/hongchenye/train_output/ckpt",
    "/user/hongchenye/train_output",
    "/backup/user/hongchenye/train/ckpt",
]

DEFAULTS = {
    "interval": 120,            # 状态检查间隔(秒)
    "report_interval": 600,     # 正常时进度播报间隔(秒)
    "stall": 1200,              # 多久无更新判定卡死(秒)
    "startup_grace": 3600,      # 启动宽限期(秒)
    "use_ct": False,
    "ct_cmd": "ct",
    "workspace": None,
}


# ----------------------------- 基础工具 -----------------------------

def now_ts() -> float:
    return time.time()


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, data: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def log_line(msg: str) -> None:
    line = f"[{now_str()}] {msg}"
    with LOGFILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None or (isinstance(seconds, float) and (math.isinf(seconds) or math.isnan(seconds))) or seconds < 0:
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


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    cfg["search_roots"] = list(DEFAULT_SEARCH_ROOTS)
    cfg["feishu_webhook"] = None
    file_cfg = load_json(CONFIG_FILE, {})
    for k, v in file_cfg.items():
        cfg[k] = v
    return cfg


# ----------------------------- 监控表 (watchlist.txt) -----------------------------

def read_watchlist() -> list:
    """返回 [(job_id, name), ...]。行格式: ``<job_id> <name...>``, # 开头为注释。"""
    entries = []
    if not WATCHLIST.exists():
        return entries
    for raw in WATCHLIST.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        job_id = parts[0]
        name = parts[1].strip() if len(parts) > 1 else ""
        entries.append((job_id, name))
    return entries


def write_watchlist(entries: list) -> None:
    lines = ["# 监控表: 每行 `<job_id> <任务名>` (名字可选)。watchdog 会自动读取。"]
    for job_id, name in entries:
        lines.append(f"{job_id} {name}".rstrip())
    WATCHLIST.write_text("\n".join(lines) + "\n", encoding="utf-8")


def upsert_entry(job_id: str, name: str) -> None:
    entries = read_watchlist()
    out = [(jid, nm) for jid, nm in entries if jid != job_id]
    out.append((job_id, name))
    write_watchlist(out)


def remove_from_watchlist(job_id: str) -> bool:
    entries = read_watchlist()
    out = [(jid, nm) for jid, nm in entries if jid != job_id]
    if len(out) != len(entries):
        write_watchlist(out)
        return True
    return False


def load_extra() -> dict:
    return load_json(EXTRA_FILE, {})


def save_extra(d: dict) -> None:
    save_json(EXTRA_FILE, d)


# ----------------------------- 自动发现配置 -----------------------------

def get_search_roots(cfg: dict) -> list:
    return cfg.get("search_roots") or list(DEFAULT_SEARCH_ROOTS)


def _iter_config_paths(roots: list):
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


# ----------------------------- ct 平台信息 (可选) -----------------------------

def ct_available(ct_cmd: str) -> bool:
    return bool(shutil.which(ct_cmd) or (os.path.exists(ct_cmd)))


def ct_job_info(job_id: str, ct_cmd: str, workspace: Optional[str]) -> Optional[dict]:
    """本机若有 ct, 返回权威状态 + 资源信息; 否则 None。"""
    exe = shutil.which(ct_cmd) or (ct_cmd if os.path.exists(ct_cmd) else None)
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [exe, "--fields",
             "id,status,cpu_num,gpu_num,memory_size,started_at,ended_at,resource_pool,train_type",
             "job", "get", str(job_id)],
            cwd=workspace or None, text=True, capture_output=True, timeout=60,
        )
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout).get("data", {})
    except Exception:
        return None


# ----------------------------- 统一报告模型 -----------------------------

# 状态常量
S_WARMUP = "WarmingUp"
S_RUNNING = "Running"
S_STALLED = "Stalled"
S_FAILED = "Failed"
S_KILLED = "Killed"
S_COMPLETED = "Completed"
S_UNKNOWN = "Unknown"
ABNORMAL = {S_STALLED, S_FAILED, S_KILLED}

STATUS_DESC = {
    S_WARMUP: "启动中(加载/编译, 暂无进度)",
    S_RUNNING: "运行中",
    S_STALLED: "长时间无更新, 疑似卡死/被杀",
    S_FAILED: "失败 Failed",
    S_KILLED: "被终止 Killed",
    S_COMPLETED: "已完成",
    S_UNKNOWN: "未知(本机无法获取, 需 ct/日志)",
}


@dataclass
class JobReport:
    job_id: str
    name: str
    kind: str = "unknown"        # training | generic | unknown
    status: str = S_UNKNOWN
    # 进度
    step: Optional[int] = None
    max_steps: Optional[int] = None
    pct: Optional[float] = None
    phase: Optional[str] = None
    speed_desc: Optional[str] = None
    eta_sec: Optional[float] = None
    finish_at: Optional[str] = None
    next_save_step: Optional[int] = None
    latest_ckpt_step: Optional[int] = None
    losses: dict = field(default_factory=dict)
    # 资源 (多数需 ct)
    gpu: Optional[Any] = None
    cpu: Optional[Any] = None
    mem: Optional[Any] = None
    pool: Optional[str] = None
    # 时间
    start_ts: Optional[float] = None
    elapsed_sec: Optional[float] = None
    freshness_sec: Optional[float] = None
    note: str = ""


def _default_name(job_id: str, start_ts: Optional[float]) -> str:
    if start_ts:
        return datetime.fromtimestamp(start_ts).strftime("%m%d_%H%M")
    return f"job_{job_id}"


def latest_ckpt_step(ckpt_dir: Optional[str]) -> Optional[int]:
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return None
    steps = []
    for p in glob.glob(os.path.join(ckpt_dir, "job_*_ckpt_*")) + glob.glob(os.path.join(ckpt_dir, "global_step*")):
        m = re.search(r"(?:_ckpt_|global_step)(\d+)$", p)
        if m:
            steps.append(int(m.group(1)))
    return max(steps) if steps else None


def _determine_training_status(rep: JobReport, now: float, stall: int, startup_grace: int) -> str:
    if rep.step is not None and rep.max_steps and rep.step >= rep.max_steps:
        return S_COMPLETED
    if rep.step is None:  # 还没有 TB 数据
        age = (now - rep.start_ts) if rep.start_ts else 0
        if age > startup_grace and (rep.freshness_sec or 0) > stall:
            return S_STALLED
        return S_WARMUP
    if (rep.freshness_sec or 0) > stall:
        return S_STALLED
    return S_RUNNING


def _cached_discover(job_id: str, roots: list, disc_cache: Optional[dict], now: float) -> Optional[str]:
    """带缓存的发现: 命中(config路径)永久缓存; 未命中每 300s 才重试一次, 避免频繁扫 /backup。"""
    if disc_cache is None:
        return discover_config_by_job_id(job_id, roots)
    ent = disc_cache.get(job_id)
    if ent is not None:
        val, ts = ent
        if val:  # 已找到, 永久用
            return val
        if now - ts < 300:  # 最近刚扫过还没找到, 先不重复扫
            return None
    val = discover_config_by_job_id(job_id, roots)
    disc_cache[job_id] = (val, now)
    return val


def build_report(job_id: str, name: str, cfg: dict, tailers: dict, now: float,
                 disc_cache: Optional[dict] = None) -> JobReport:
    """核心: 为一个 job 构造统一报告。谁能填就填。"""
    roots = get_search_roots(cfg)
    extra = load_extra().get(job_id, {})
    stall = cfg["stall"]
    startup_grace = cfg["startup_grace"]

    rep = JobReport(job_id=job_id, name=name)

    # 1) 训练任务: 自动发现 all_config.json
    config_path = extra.get("config") or _cached_discover(job_id, roots, disc_cache, now)
    if config_path and os.path.exists(config_path):
        try:
            ta = json.load(open(config_path, encoding="utf-8")).get("training_args", {})
        except Exception:
            ta = {}
        tb_dir = ta.get("tensorboard")
        rep.kind = "training"
        rep.max_steps = ta.get("max_steps")
        rep.gpu = ta.get("world_size") or None      # 训练卡数 = world_size
        ckpt_dir = ta.get("exp_ckpt_dir")
        rep.start_ts = parse_start_ts_from_path(tb_dir or "")

        if tb_dir:
            tailer = tailers.get(job_id)
            if tailer is None or tailer.tb_dir != tb_dir:
                tailer = TBTailer(tb_dir=tb_dir)
                tailers[job_id] = tailer
            try:
                tailer.poll()
            except Exception as e:  # noqa: BLE001
                log_line(f"[{job_id}] 读取 TB 失败: {e}")
            if tailer.latest_step >= 0:
                rep.step = tailer.latest_step
                rep.pct = (rep.step / rep.max_steps) if rep.max_steps else None
                sps = tailer.steps_per_sec()
                if sps:
                    spstep = 1.0 / sps
                    rep.speed_desc = f"{spstep:.1f}秒/step (~{3600/spstep:.0f} step/时)"
                    if rep.max_steps and rep.step < rep.max_steps:
                        rep.eta_sec = (rep.max_steps - rep.step) / sps
                        rep.finish_at = (datetime.now() + timedelta(seconds=rep.eta_sec)).strftime("%m-%d %H:%M")
                if ta.get("warmup_t") and rep.step <= ta["warmup_t"]:
                    rep.phase = f"warmup 预热 ({rep.step}/{ta['warmup_t']})"
                else:
                    rep.phase = "正式训练"
                if ta.get("save_step") and rep.step < (rep.max_steps or 0):
                    rep.next_save_step = min(((rep.step // ta["save_step"]) + 1) * ta["save_step"], rep.max_steps)
                rep.losses = dict(tailer.losses)
            mtime = tailer.event_file_mtime()
            rep.freshness_sec = (now - mtime) if mtime else None
            rep.latest_ckpt_step = latest_ckpt_step(ckpt_dir)

        rep.elapsed_sec = (now - rep.start_ts) if rep.start_ts else None
        rep.status = _determine_training_status(rep, now, stall, startup_grace)

    # 2) 通用任务: 指定了日志路径
    elif extra.get("log"):
        rep.kind = "generic"
        logp = extra["log"]
        if os.path.exists(logp):
            mtime = os.path.getmtime(logp)
            rep.freshness_sec = now - mtime
            rep.start_ts = rep.start_ts or os.path.getctime(logp)
            rep.elapsed_sec = (now - rep.start_ts) if rep.start_ts else None
            rep.status = S_STALLED if rep.freshness_sec > stall else S_RUNNING
            prog = _scan_log_progress(logp, extra.get("progress_regex"))
            if prog:
                rep.step, rep.max_steps = prog
                rep.pct = (rep.step / rep.max_steps) if rep.max_steps else None
        else:
            rep.note = f"日志文件不存在: {logp}"

    else:
        rep.kind = "unknown"
        rep.status = S_UNKNOWN
        rep.note = "本机无 ct/平台接口, 且未提供日志路径, 无法获取该任务状态/CPU/GPU。"

    # 3) ct 增强 (若本机有 ct): 覆盖状态 + 补资源, 对任意任务类型都生效
    if cfg.get("use_ct"):
        info = ct_job_info(job_id, cfg.get("ct_cmd", "ct"), cfg.get("workspace"))
        if info:
            rep.cpu = info.get("cpu_num", rep.cpu)
            rep.gpu = info.get("gpu_num", rep.gpu)
            rep.mem = info.get("memory_size", rep.mem)
            rep.pool = info.get("resource_pool", rep.pool)
            st = info.get("status")
            mapping = {"Running": S_RUNNING, "Failed": S_FAILED, "Killed": S_KILLED,
                       "Succeeded": S_COMPLETED}
            if st in mapping and rep.kind != "training":
                rep.status = mapping[st]
            elif st in ("Failed", "Killed"):
                rep.status = mapping[st]  # 训练任务也尊重 ct 的失败/终止
            if info.get("started_at"):
                rep.note = (rep.note + " ").strip()

    # 缺省名字: 用启动时间
    if not rep.name:
        rep.name = _default_name(job_id, rep.start_ts)
    return rep


def extract_log_from_cmd(cmd: str) -> Optional[str]:
    """从一条完整启动命令里抽取日志文件路径 (处理相对路径 + 前面的 cd)。

    支持:  ``>> a.log`` / ``> a.log`` / ``&> a.log`` / ``2>&1``(忽略) / ``tee [-a] a.log``。
    相对路径会用重定向之前最后一个 ``cd <dir>`` 拼成绝对路径。
    """
    redir_re = re.compile(r"(?:&>>?|1?>>?)\s*([^\s&|;<>]+)")
    tee_re = re.compile(r"\btee\b\s+(?:-a\s+)?([^\s&|;<>]+)")
    candidates = []  # (pos, target)
    for m in redir_re.finditer(cmd):
        tgt = m.group(1)
        if tgt in ("/dev/null",) or tgt.startswith("&"):
            continue
        candidates.append((m.start(), tgt))
    for m in tee_re.finditer(cmd):
        candidates.append((m.start(), m.group(1)))
    if not candidates:
        return None
    candidates.sort()
    chosen = next((c for c in candidates if c[1].endswith(".log")), candidates[0])
    pos, target = chosen
    if os.path.isabs(target):
        return os.path.normpath(target)
    # 找该重定向之前最后一个 cd DIR 作为 base
    base = None
    for m in re.finditer(r"\bcd\s+([^\s&|;]+)", cmd):
        if m.start() < pos and os.path.isabs(m.group(1)):
            base = m.group(1)
    if base:
        return os.path.normpath(os.path.join(base, target))
    return target


def _scan_log_progress(path: str, regex: Optional[str]) -> Optional[tuple]:
    """从日志尾部扫描形如 X/Y 的进度; 找不到返回 None。"""
    pat = re.compile(regex) if regex else re.compile(r"(\d+)\s*/\s*(\d+)")
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 8192))
            tail = f.read().decode("utf-8", "replace")
    except Exception:
        return None
    last = None
    for m in pat.finditer(tail):
        try:
            a, b = int(m.group(1)), int(m.group(2))
            if b > 0 and a <= b:
                last = (a, b)
        except Exception:
            continue
    return last


# ----------------------------- 渲染 -----------------------------

def render_card(rep: JobReport) -> str:
    lines = []
    lines.append(f"**状态**: {rep.status} — {STATUS_DESC.get(rep.status, '')}")
    kind_cn = {"training": "训练任务", "generic": "通用任务", "unknown": "未知类型"}.get(rep.kind, rep.kind)
    lines.append(f"**类型**: {kind_cn}")

    # 资源
    res = []
    if rep.gpu not in (None, 0, "0"):
        res.append(f"GPU {rep.gpu}卡")
    if rep.cpu not in (None, ""):
        res.append(f"CPU {rep.cpu}")
    if rep.mem not in (None, ""):
        res.append(f"内存 {rep.mem}")
    if rep.pool:
        res.append(f"池 {rep.pool}")
    if res:
        lines.append("**资源**: " + "  ".join(str(x) for x in res))

    # 进度
    if rep.step is not None and rep.max_steps:
        lines.append(f"**进度**: {rep.step}/{rep.max_steps} ({rep.pct*100:.1f}%)\n`{progress_bar(rep.pct or 0)}`")
        if rep.phase:
            lines.append(f"**阶段**: {rep.phase}")
        if rep.next_save_step:
            extra = f" (已存到 {rep.latest_ckpt_step})" if rep.latest_ckpt_step else ""
            lines.append(f"**下次存 ckpt**: step {rep.next_save_step}{extra}")
        if rep.speed_desc:
            lines.append(f"**速度**: {rep.speed_desc}")

    if rep.elapsed_sec is not None:
        lines.append(f"**已运行**: {fmt_duration(rep.elapsed_sec)}")
    if rep.eta_sec is not None and rep.status != S_COMPLETED:
        tail = f" (约 {rep.finish_at} 完成)" if rep.finish_at else ""
        lines.append(f"**预计剩余**: {fmt_duration(rep.eta_sec)}{tail}")
    if rep.losses:
        loss_str = "  ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in rep.losses.items())
        lines.append(f"**Loss**: {loss_str}")
    if rep.freshness_sec is not None:
        lines.append(f"**数据更新**: {fmt_duration(rep.freshness_sec)}前")
    lines.append(f"**检查时间**: {now_str()}")
    if rep.note:
        lines.append(f"> {rep.note}")
    return "\n".join(lines)


def card_title(rep: JobReport) -> str:
    return f"watchdog播报：任务 {rep.job_id}（{rep.name}）"


def level_for(status: str) -> str:
    if status in ABNORMAL:
        return "alert"
    if status == S_COMPLETED:
        return "ok"
    return "info"


# ----------------------------- 每 job 通知状态 -----------------------------

def job_state_path(job_id: str) -> Path:
    return STATE_DIR / f"{job_id}.json"


def handle_notifications(rep: JobReport, notifier: FeishuNotifier, report_interval: int, now: float) -> None:
    sp = job_state_path(rep.job_id)
    st = load_json(sp, {})
    started = st.get("started_notified", False)
    last_report = st.get("last_report_ts", 0)
    incident = st.get("incident")
    completed = st.get("completed_notified", False)

    title = card_title(rep)
    body = render_card(rep)

    if not started:
        notifier.send(title, "🟦 已开始监控该任务。\n\n" + body, level="info")
        started = True
        last_report = now
    elif rep.status == S_COMPLETED:
        if not completed:
            notifier.send(title, "✅ 训练已完成。\n\n" + body, level="ok")
            completed = True
        incident = None
    elif rep.status in ABNORMAL:
        if incident != rep.status:
            hint = "\n> ⚠️ 请人工确认 (查看 job 状态 / 节点)。" if rep.status == S_STALLED else ""
            notifier.send(title, "🚨 任务异常告警！\n\n" + body + hint, level="alert")
            incident = rep.status
    else:  # 正常
        if incident is not None:
            notifier.send(title, "🟢 已从异常恢复。\n\n" + body, level="ok")
            incident = None
            last_report = now
        elif rep.status in (S_RUNNING,) and (now - last_report) >= report_interval:
            notifier.send(title, "📊 进度播报\n\n" + body, level="info")
            last_report = now
        elif rep.status == S_UNKNOWN and (now - last_report) >= report_interval:
            # 未知任务也按节奏播报一次现状(不告警)
            notifier.send(title, "ℹ️ 现状\n\n" + body, level="info")
            last_report = now

    save_json(sp, {
        "started_notified": started, "last_report_ts": last_report,
        "incident": incident, "completed_notified": completed,
        "last_status": rep.status, "name": rep.name, "kind": rep.kind,
        "step": rep.step, "max_steps": rep.max_steps, "pct": rep.pct,
        "eta_sec": rep.eta_sec, "updated_at": now_str(),
    })


# ----------------------------- 全表快照 -----------------------------

def write_status_md(reports: list) -> None:
    lines = ["# Watchdog 监控表快照", "", f"更新时间: {now_str()}", ""]
    lines.append("| Job | 名称 | 类型 | 状态 | 进度 | 已运行 | 预计剩余 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in reports:
        prog = f"{r.step}/{r.max_steps} ({r.pct*100:.1f}%)" if (r.step is not None and r.max_steps) else "-"
        lines.append(f"| {r.job_id} | {r.name} | {r.kind} | {r.status} | {prog} | "
                     f"{fmt_duration(r.elapsed_sec)} | {fmt_duration(r.eta_sec) if r.eta_sec else '-'} |")
    lines.append("")
    for r in reports:
        lines.append(f"## {r.job_id} · {r.name}\n")
        lines.append(render_card(r))
        lines.append("")
    STATUS_MD.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------- 守护进程主循环 -----------------------------

def serve(args: argparse.Namespace) -> None:
    cfg = load_config()
    # 命令行覆盖
    for k in ("interval", "report_interval", "stall", "startup_grace"):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    if getattr(args, "use_ct", False):
        cfg["use_ct"] = True

    PIDFILE.write_text(str(os.getpid()), encoding="utf-8")
    webhook = getattr(args, "webhook", None) or cfg.get("feishu_webhook") or load_webhook_from_config(str(CONFIG_FILE))
    notifier = FeishuNotifier(webhook=webhook, logger=log_line)

    log_line(f"watchdog 守护进程启动 pid={os.getpid()} "
             f"interval={cfg['interval']}s report={cfg['report_interval']}s "
             f"stall={cfg['stall']}s notify={'on' if notifier.enabled else 'off'} "
             f"ct={'on' if cfg.get('use_ct') and ct_available(cfg.get('ct_cmd','ct')) else 'off'}")

    tailers: dict = {}
    disc_cache: dict = {}
    while True:
        now = now_ts()
        entries = read_watchlist()
        reports = []
        seen = set()
        for job_id, name in entries:
            seen.add(job_id)
            try:
                rep = build_report(job_id, name, cfg, tailers, now, disc_cache)
                handle_notifications(rep, notifier, cfg["report_interval"], now)
                reports.append(rep)
                append_jsonl(EVENTS, {"time": now_str(), "job": job_id, "name": rep.name,
                                      "status": rep.status, "step": rep.step, "pct": rep.pct})
                log_line(f"[{job_id}] {rep.name} status={rep.status} "
                         f"step={rep.step}/{rep.max_steps} fresh={fmt_duration(rep.freshness_sec)}")
            except Exception as e:  # noqa: BLE001
                log_line(f"[{job_id}] 评估失败: {e}")
        # 清掉已移除 job 的 tailer 缓存
        for jid in list(tailers.keys()):
            if jid not in seen:
                tailers.pop(jid, None)
        try:
            write_status_md(reports)
        except Exception as e:  # noqa: BLE001
            log_line(f"写 status.md 失败: {e}")
        time.sleep(cfg["interval"])


def daemon_alive() -> Optional[int]:
    if not PIDFILE.exists():
        return None
    pid = PIDFILE.read_text().strip()
    if pid.isdigit() and os.path.exists(f"/proc/{pid}"):
        return int(pid)
    return None


# ----------------------------- CLI -----------------------------

def cmd_start(args: argparse.Namespace) -> None:
    pid = daemon_alive()
    if pid and not args.foreground:
        print(f"watchdog 已在运行 (pid={pid})。如需改参数: watchdog stop 后再 start, 或 watchdog restart。")
        return
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if getattr(args, "foreground", False):
        serve(args)
        return
    cmd = [sys.executable, str(Path(__file__).resolve()), "_serve"]
    for k in ("interval", "report_interval", "stall", "startup_grace"):
        v = getattr(args, k, None)
        if v is not None:
            cmd += [f"--{k.replace('_', '-')}", str(v)]
    if args.use_ct:
        cmd += ["--use-ct"]
    if args.webhook:
        cmd += ["--webhook", args.webhook]
    out = LOGFILE.open("a", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT,
                            start_new_session=True)
    PIDFILE.write_text(str(proc.pid), encoding="utf-8")
    print(f"watchdog 已启动 (pid={proc.pid})。")
    print(f"监控表: {WATCHLIST}")
    print(f"全表快照: {STATUS_MD}")
    print(f"日志: {LOGFILE}")


def cmd_stop(args: argparse.Namespace) -> None:
    pid = daemon_alive()
    if not pid:
        print("watchdog 未在运行。")
        return
    try:
        os.kill(pid, 15)
        print(f"已停止 watchdog (pid={pid})。")
    except ProcessLookupError:
        print("进程已不存在。")


def cmd_restart(args: argparse.Namespace) -> None:
    cmd_stop(args)
    time.sleep(1)
    cmd_start(args)


def cmd_add(args: argparse.Namespace) -> None:
    cfg = load_config()
    job_id = str(args.job_id)
    name = " ".join(args.name).strip() if args.name else ""

    # 若给了完整启动命令, 自动抽取日志路径
    log_path = args.log
    if not log_path and args.cmd:
        log_path = extract_log_from_cmd(args.cmd)
        if log_path:
            print(f"[{job_id}] 从启动命令解析出日志: {log_path}")
        else:
            print(f"[{job_id}] 未能从命令里解析出日志路径, 请改用 --log 手动指定。")

    # 记录高级配置(日志路径/手动 config/命令)
    if log_path or args.config or args.cmd:
        extra = load_extra()
        e = extra.get(job_id, {})
        if log_path:
            e["log"] = log_path
        if args.config:
            e["config"] = args.config
        if args.cmd:
            e["cmd"] = args.cmd
        extra[job_id] = e
        save_extra(extra)

    cfg_path = args.config or discover_config_by_job_id(job_id, get_search_roots(cfg))
    # 缺省名字: 用启动时间(能发现的话)
    if not name:
        start_ts = None
        if cfg_path and os.path.exists(cfg_path):
            try:
                tb = json.load(open(cfg_path, encoding="utf-8")).get("training_args", {}).get("tensorboard", "")
                start_ts = parse_start_ts_from_path(tb)
            except Exception:
                pass
        name = _default_name(job_id, start_ts)

    upsert_entry(job_id, name)
    kind = "训练任务" if cfg_path else ("通用任务(日志)" if log_path else "未知(需 ct/日志)")
    print(f"[{job_id}] 已加入监控表, 名称='{name}', 识别为 {kind}。")
    if not daemon_alive():
        print("提示: watchdog 守护进程未运行, 用  python watchdog.py start  开启。")


def cmd_rm(args: argparse.Namespace) -> None:
    for job_id in args.job_ids:
        job_id = str(job_id)
        ok = remove_from_watchlist(job_id)
        extra = load_extra()
        if job_id in extra:
            extra.pop(job_id)
            save_extra(extra)
        if args.purge:
            sp = job_state_path(job_id)
            if sp.exists():
                sp.unlink()
        print(f"[{job_id}] " + ("已从监控表移除。" if ok else "不在监控表中。"))


def cmd_ls(args: argparse.Namespace) -> None:
    entries = read_watchlist()
    pid = daemon_alive()
    print(f"守护进程: {'运行中 pid=' + str(pid) if pid else '未运行'}   监控表: {WATCHLIST}")
    if not entries:
        print("监控表为空。用  python watchdog.py add <job_id> [名字]  添加。")
        return
    header = f"{'JOB':<8} {'NAME':<16} {'STATUS':<10} {'STEP':<12} {'PCT':<7} {'ETA':<12} LAST"
    print(header)
    print("-" * len(header))
    for job_id, name in entries:
        st = load_json(job_state_path(job_id), {})
        status = st.get("last_status", "-")
        step = st.get("step")
        mx = st.get("max_steps")
        step_s = f"{step}/{mx}" if (step is not None and mx) else "-"
        pct = st.get("pct")
        pct_s = f"{pct*100:.1f}%" if isinstance(pct, (int, float)) else "-"
        eta = st.get("eta_sec")
        eta_s = fmt_duration(eta) if isinstance(eta, (int, float)) else "-"
        print(f"{job_id:<8} {(name or st.get('name') or '-'):<16} {status:<10} "
              f"{step_s:<12} {pct_s:<7} {eta_s:<12} {st.get('updated_at','-')}")
    print("\n(详细进度看:  python watchdog.py show <job_id>  或  cat status.md)")


def cmd_show(args: argparse.Namespace) -> None:
    cfg = load_config()
    if cfg.get("use_ct"):
        pass
    # 直接现算一次, 保证最新
    tailers: dict = {}
    entries = dict(read_watchlist())
    job_id = str(args.job_id)
    name = entries.get(job_id, "")
    rep = build_report(job_id, name, cfg, tailers, now_ts())
    print(card_title(rep))
    print(render_card(rep))


def cmd_once(args: argparse.Namespace) -> None:
    cfg = load_config()
    if args.use_ct:
        cfg["use_ct"] = True
    tailers: dict = {}
    entries = read_watchlist()
    if args.job_id:
        entries = [(str(args.job_id), dict(entries).get(str(args.job_id), ""))]
    now = now_ts()
    reports = []
    for job_id, name in entries:
        rep = build_report(job_id, name, cfg, tailers, now)
        reports.append(rep)
        print(card_title(rep))
        print(render_card(rep))
        print("-" * 60)
        if args.send:
            webhook = cfg.get("feishu_webhook") or load_webhook_from_config(str(CONFIG_FILE))
            FeishuNotifier(webhook=webhook).send(card_title(rep), render_card(rep), level=level_for(rep.status))


def cmd_test_notify(args: argparse.Namespace) -> None:
    cfg = load_config()
    webhook = args.webhook or cfg.get("feishu_webhook") or load_webhook_from_config(str(CONFIG_FILE))
    n = FeishuNotifier(webhook=webhook)
    if not n.enabled:
        print("未配置 webhook。请在 config.json 填 feishu_webhook, 或用 --webhook。")
        return
    ok = n.send("watchdog播报：测试消息", f"这是一条测试消息。\n**时间**: {now_str()}", level="ok")
    print("发送成功。" if ok else "发送失败, 请检查 webhook。")


def _add_service_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--interval", type=int, help="状态检查间隔(秒)")
    p.add_argument("--report-interval", type=int, help="正常时进度播报间隔(秒)")
    p.add_argument("--stall", type=int, help="多久无更新判定卡死(秒)")
    p.add_argument("--startup-grace", type=int, help="启动宽限期(秒)")
    p.add_argument("--use-ct", action="store_true", help="用 ct 拿权威状态/资源(本机需有 ct)")
    p.add_argument("--webhook", help="飞书 webhook")


def main() -> int:
    parser = argparse.ArgumentParser(description="训练/任务 watchdog (监控表 + 单守护进程 + 飞书播报)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser("start", help="开启 watchdog 守护进程")
    _add_service_args(p_start)
    p_start.add_argument("--foreground", action="store_true", help="前台运行(调试)")
    p_start.set_defaults(func=cmd_start)

    p_stop = sub.add_parser("stop", help="关闭 watchdog 守护进程")
    p_stop.set_defaults(func=cmd_stop)

    p_restart = sub.add_parser("restart", help="重启守护进程(用于改参数)")
    _add_service_args(p_restart)
    p_restart.set_defaults(func=cmd_restart)

    p_add = sub.add_parser("add", help="加 job 到监控表 (只需 job id, 可带名字)")
    p_add.add_argument("job_id", help="job id")
    p_add.add_argument("name", nargs="*", help="任务名字(可选, 缺省用启动时间)")
    p_add.add_argument("--log", help="通用任务的日志文件路径(非训练任务用)")
    p_add.add_argument("--cmd", help="完整启动命令(自动从中抽取日志路径, 非训练任务用)")
    p_add.add_argument("--config", help="手动指定 all_config.json")
    p_add.set_defaults(func=cmd_add)

    p_rm = sub.add_parser("rm", help="从监控表移除")
    p_rm.add_argument("job_ids", nargs="+")
    p_rm.add_argument("--purge", action="store_true", help="同时清除该 job 的通知状态")
    p_rm.set_defaults(func=cmd_rm)

    p_ls = sub.add_parser("ls", help="查看监控表")
    p_ls.set_defaults(func=cmd_ls)

    p_show = sub.add_parser("show", help="现算并打印某 job 的详细快照")
    p_show.add_argument("job_id")
    p_show.set_defaults(func=cmd_show)

    p_once = sub.add_parser("once", help="现算一次(全部或单个)并打印")
    p_once.add_argument("job_id", nargs="?")
    p_once.add_argument("--send", action="store_true")
    p_once.add_argument("--use-ct", action="store_true")
    p_once.set_defaults(func=cmd_once)

    p_test = sub.add_parser("test-notify", help="发一条飞书测试消息")
    p_test.add_argument("--webhook")
    p_test.set_defaults(func=cmd_test_notify)

    p_serve = sub.add_parser("_serve", help="(内部)守护进程主循环")
    _add_service_args(p_serve)
    p_serve.set_defaults(func=serve)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
