#!/usr/bin/env python3
"""从 TensorBoard event 文件增量读取训练进度指标。

设计要点:
- TensorBoard 的 event 文件是 TFRecord 格式, 训练进程持续往里追加写入。
- 我们记录已读取的字节偏移 (offset), 每次只解析新增的记录, 避免每轮全量解析 (文件会涨到几百 MB)。
- 只关心 scalar summary (simple_value), 从中取出当前 step / wall_time / 关键 loss。

对外主要接口:
- ``find_latest_event_file(tb_dir)``
- ``TBTailer``  : 有状态的增量读取器, 维护 offset 与关键指标。
"""

from __future__ import annotations

import glob
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Optional

# tensorboard 一定装了 (训练环境自带); 只用它的 protobuf 定义, 不依赖 TF。
from tensorboard.compat.proto import event_pb2  # type: ignore


# 训练脚本记录到 TB 的关键标量 (见 all_config.json: log_tts_loss / enable_distill 等)
LOSS_TAGS = ["Loss/train", "Loss/ce", "Loss/kl", "Loss/tts_loss"]


def find_latest_event_file(tb_dir: str) -> Optional[str]:
    """返回 tb_dir (可含子目录) 下最新的 events.out.tfevents.* 文件。"""
    if not tb_dir or not os.path.isdir(tb_dir):
        return None
    candidates = glob.glob(os.path.join(tb_dir, "**", "events.out.tfevents.*"), recursive=True)
    candidates += glob.glob(os.path.join(tb_dir, "events.out.tfevents.*"))
    candidates = list(set(candidates))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]


def parse_start_ts_from_path(path: str) -> Optional[float]:
    """从形如 ``...-job_141369-20260708071107`` 的路径里解析训练启动时间 (本地时间戳)。"""
    m = re.search(r"(\d{14})", path)
    if not m:
        return None
    import datetime as _dt

    try:
        dt = _dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
        return dt.timestamp()
    except Exception:
        return None


def parse_job_id_from_path(path: str) -> Optional[str]:
    m = re.search(r"job[_-](\d{3,})", path)
    return m.group(1) if m else None


@dataclass
class TBSample:
    wall: float
    step: int


@dataclass
class TBTailer:
    """增量跟踪单个 TB run 的进度。跨轮次复用同一个实例即可。"""

    tb_dir: str
    max_step_window: int = 200  # 保留最近 N 个 (wall, step) 采样用于估算速度

    # 内部状态
    event_file: Optional[str] = None
    offset: int = 0
    latest_step: int = -1
    latest_wall: float = 0.0
    first_wall: float = 0.0
    first_step: int = -1
    losses: dict = field(default_factory=dict)
    samples: list = field(default_factory=list)  # List[TBSample]
    total_events: int = 0

    def _reset_for_new_file(self, path: str) -> None:
        self.event_file = path
        self.offset = 0
        # 不清空 latest_* / samples: 训练重启续跑时 step 会接着涨, 保留历史有利于速度估计。

    def poll(self) -> bool:
        """读取新增事件, 更新指标。返回是否成功拿到 (至少一个) event 文件。"""
        path = find_latest_event_file(self.tb_dir)
        if path is None:
            return False
        if path != self.event_file:
            self._reset_for_new_file(path)

        try:
            size = os.path.getsize(path)
        except OSError:
            return False
        if size < self.offset:
            # 文件被截断/重写, 从头再来
            self.offset = 0
        if size == self.offset:
            return True  # 没有新数据

        with open(path, "rb") as f:
            f.seek(self.offset)
            while True:
                pos = f.tell()
                header = f.read(8)
                if len(header) < 8:
                    f.seek(pos)
                    break
                (length,) = struct.unpack("<Q", header)
                # 4 字节 crc(len) + data + 4 字节 crc(data)
                rest = f.read(4 + length + 4)
                if len(rest) < 4 + length + 4:
                    f.seek(pos)  # 记录不完整 (writer 正在写), 下轮再读
                    break
                data = rest[4 : 4 + length]
                try:
                    ev = event_pb2.Event()
                    ev.ParseFromString(data)
                except Exception:
                    continue
                self.total_events += 1
                self._consume(ev)
            self.offset = f.tell()
        return True

    def _consume(self, ev) -> None:
        if not ev.HasField("summary"):
            return
        vals = {}
        for v in ev.summary.value:
            if v.HasField("simple_value"):
                vals[v.tag] = v.simple_value
        if not vals:
            return
        step = int(ev.step)
        wall = float(ev.wall_time)
        if self.first_step < 0 or step < self.first_step:
            self.first_step = step
            self.first_wall = wall
        if step >= self.latest_step:
            self.latest_step = step
            self.latest_wall = wall
            for k in LOSS_TAGS:
                if k in vals:
                    self.losses[k] = vals[k]
        # 速度采样 (只在 step 前进时记录, 去重)
        if not self.samples or self.samples[-1].step != step:
            self.samples.append(TBSample(wall=wall, step=step))
            if len(self.samples) > self.max_step_window:
                self.samples.pop(0)

    # ---- 派生指标 ----
    def steps_per_sec(self) -> Optional[float]:
        """基于采样窗口估算训练速度 (steps/sec)。"""
        pts = [s for s in self.samples if s.step >= 0]
        if len(pts) < 2:
            return None
        a, b = pts[0], pts[-1]
        dstep = b.step - a.step
        dwall = b.wall - a.wall
        if dstep <= 0 or dwall <= 0:
            return None
        return dstep / dwall

    def event_file_mtime(self) -> Optional[float]:
        if self.event_file and os.path.exists(self.event_file):
            return os.path.getmtime(self.event_file)
        return None
