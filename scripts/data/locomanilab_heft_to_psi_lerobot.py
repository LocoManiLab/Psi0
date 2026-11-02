"""Convert a LocoManiLab-prepared HEFT LeRobot dataset into the Psi0 training format.

Sibling of locomanilab_sonic_to_psi_lerobot.py for the OTHER low-level G1 controller.
heft (external/heft, GMR-retargeted motion tracking) records a different thing than
sonic: the action IS the tracking policy's reference frame, not a learned latent.

  sonic:  state 43 (incl. hands)   action 78 = motion_token64 + hands14
  heft:   state 29 (no hands)      action 50 = ref_qpos36 (root_pos3+quat4+joint29)
                                              + hands14 (Dex3, 7 per hand)

So unlike the sonic converter there is no state re-ordering to do — heft's 29-D state
is already qpos order (lower15 + larm7 + rarm7) and carries no hand columns. Train with
matching dims: `--set model.action-dim=50 --set model.odim=29`.

Output layout (video key, meta, chunking) is identical to the sonic converter.

Usage:
  python scripts/data/locomanilab_heft_to_psi_lerobot.py \
      --data-root=/path/to/datasets/<task> \
      --work-dir=$PSI_HOME/data/sonic/lerobot --repo-id=<task>
"""
import argparse
import json
import logging
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, Features, Sequence, Value
from datasets.utils.logging import set_verbosity_error
from tqdm import tqdm

CODE_VERSION = "v2.1"

set_verbosity_error()
logging.getLogger("pyarrow").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

# --- source (lab prepare) column names ---
SRC_VIDEO_KEY = "observation.images.ego_view"
SRC_STATE = "observation.state"                    # 29: lower15 + larm7 + rarm7
SRC_REF_QPOS = "action.ref_qpos"                   # 36: root_pos3 + root_quat4 + joint29
SRC_HANDS = "action.hands"                         # 14: Dex3 left7 + right7

STATE_DIM, ACTION_DIM = 29, 50



@dataclass
class InfoDict:
    codebase_version: str
    robot_type: str
    total_episodes: int
    total_frames: int
    total_tasks: int
    total_videos: int
    total_chunks: int
    chunks_size: int
    fps: int
    data_path: str
    video_path: str
    features: Dict[str, Any]


def append_jsonl_line_atomic(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n"
    fd = os.open(str(path), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "a", encoding="utf-8") as f:
        try:
            import fcntl

            fcntl.flock(f, fcntl.LOCK_EX)
        except Exception:
            pass
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def take_slices(vec: np.ndarray, slices: List[Tuple[int, int]]) -> np.ndarray:
    return np.concatenate([vec[a:b] for a, b in slices])


class LocoManiLabHeft2PsiConverter:
    """LocoManiLab-prepared heft LeRobot -> Psi0 LeRobot (states 29 / action 50)."""

    def __init__(self, fps: float, video_shape: List[int]):
        self.fps = fps
        self.video_shape = video_shape
        self.features = Features(
            {
                "states": Sequence(Value("float32")),
                "action": Sequence(Value("float32")),
                "timestamp": Value("float32"),
                "frame_index": Value("int64"),
                "episode_index": Value("int64"),
                "index": Value("int64"),
                "task_index": Value("int64"),
                "next.done": Value("bool"),
            }
        )
        self.tasks_meta: Dict[int, str] = {}
        self.episode_sources: List[Tuple[int, Path, Path, int]] = []
        self.lengths_by_episode: Dict[int, int] = {}
        self.chunks_size: int = 1000

    def build_obs(self, state29: np.ndarray) -> Dict[str, Any]:
        # heft state is already in qpos order and has no hand columns — pass through
        return {"states": state29.astype(np.float32).tolist()}   # 29

    def build_act(self, ref_qpos36: np.ndarray, hands14: np.ndarray) -> List[float]:
        action = np.concatenate([ref_qpos36, hands14])
        assert action.shape[0] == ACTION_DIM, f"action dim {action.shape[0]} != {ACTION_DIM}"
        return action.astype(np.float32).tolist()                # 36 + 14 = 50

    def make_one_episode(
        self,
        task_index: int,
        episode_index: int,
        src_parquet: Path,
        src_video: Path,
        out_base: Path,
        chunks_size: int,
    ) -> Tuple[int, int]:
        chunk_path = out_base / f"chunk-{episode_index // chunks_size:03d}"
        chunk_path.mkdir(parents=True, exist_ok=True)
        parquet_path = chunk_path / f"episode_{episode_index:06d}.parquet"

        ego_dir = out_base.parent / "videos" / f"chunk-{episode_index // chunks_size:03d}" / "observation.images.egocentric"
        ego_dir.mkdir(parents=True, exist_ok=True)
        vid_path = ego_dir / f"episode_{episode_index:06d}.mp4"

        df = pd.read_parquet(src_parquet)
        n = len(df)
        assert n > 0, f"empty parquet {src_parquet}"
        state = np.vstack([np.asarray(x, dtype=np.float64) for x in df[SRC_STATE]])
        refq = np.vstack([np.asarray(x, dtype=np.float64) for x in df[SRC_REF_QPOS]])
        hands = np.vstack([np.asarray(x, dtype=np.float64) for x in df[SRC_HANDS]])
        assert state.shape[1] == STATE_DIM, f"state dim {state.shape[1]} != {STATE_DIM}"
        assert refq.shape[1] == 36, f"ref_qpos dim {refq.shape[1]} != 36"
        assert hands.shape[1] == 14, f"hands dim {hands.shape[1]} != 14"

        rows: List[Dict[str, Any]] = []
        for i in range(n):
            rows.append(
                {
                    **self.build_obs(state[i]),
                    "action": self.build_act(refq[i], hands[i]),
                    "timestamp": i * (1.0 / self.fps),
                    "frame_index": i,
                    "episode_index": episode_index,
                    "index": i,
                    "task_index": task_index,
                    "next.done": (i == n - 1),
                }
            )

        tmp_dir = out_base / f"_tmp_ep_{episode_index:06d}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        parquet_tmp = tmp_dir / "episode.parquet"
        Dataset.from_list(rows, features=self.features).to_parquet(str(parquet_tmp))
        os.replace(parquet_tmp, parquet_path)
        shutil.copyfile(src_video, vid_path)  # already h264/yuv420p
        shutil.rmtree(tmp_dir, ignore_errors=True)

        # per-episode action/timestamp stats -> episodes_stats.jsonl
        acts = np.array([r["action"] for r in rows], dtype=np.float32)
        episode_stats = {
            "episode_index": episode_index,
            "stats": {
                "action": {
                    "min": acts.min(0).tolist(), "max": acts.max(0).tolist(),
                    "mean": acts.mean(0).tolist(), "std": acts.std(0).tolist(),
                    "count": [n],
                },
                "timestamp": {
                    "min": [0.0], "max": [(n - 1) / self.fps],
                    "mean": [((n - 1) / 2) / self.fps],
                    "std": [n / (2 * self.fps * math.sqrt(3))], "count": [n],
                },
            },
        }
        append_jsonl_line_atomic(out_base.parent / "meta" / "episodes_stats.jsonl", episode_stats)
        return episode_index, n

    def run(self, data_root: Path, work_dir: Path, chunks_size: int, num_workers: int):
        self.chunks_size = chunks_size
        data_dir = work_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        # rerun-safe: episodes_stats.jsonl is append-only per episode
        stats_jsonl = work_dir / "meta" / "episodes_stats.jsonl"
        if stats_jsonl.exists():
            stats_jsonl.unlink()

        task_to_idx: Dict[str, int] = {}
        for r in read_jsonl(data_root / "meta" / "tasks.jsonl"):
            task_to_idx[r["task"]] = r["task_index"]
            self.tasks_meta[r["task_index"]] = r["task"]
        ep_task: Dict[int, str] = {}
        for r in read_jsonl(data_root / "meta" / "episodes.jsonl"):
            tasks = r.get("tasks", [])
            ep_task[r["episode_index"]] = tasks[0] if tasks else ""

        self.episode_sources = []
        out_ep = 0
        for pq in sorted((data_root / "data").rglob("episode_*.parquet")):
            src_ep = int(pq.stem.split("_")[1])
            desc = ep_task.get(src_ep, "")
            task_idx = task_to_idx.get(desc, 0)
            video = (
                data_root / "videos" / f"chunk-{src_ep // chunks_size:03d}"
                / SRC_VIDEO_KEY / f"episode_{src_ep:06d}.mp4"
            )
            assert video.is_file(), f"missing source video: {video}"
            self.episode_sources.append((task_idx, pq, video, out_ep))
            out_ep += 1

        print(f"Found {len(self.episode_sources)} episodes, {len(self.tasks_meta)} tasks.")
        if not self.episode_sources:
            print("No episodes found.")
            return

        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [
                ex.submit(self.make_one_episode, task_idx, oi, pq, vid, data_dir, chunks_size)
                for (task_idx, pq, vid, oi) in self.episode_sources
            ]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing episodes", unit="ep"):
                ep_idx, n_frames = fut.result()
                self.lengths_by_episode[ep_idx] = n_frames

        self.num_episodes = len(self.lengths_by_episode)
        self.total_frames = sum(self.lengths_by_episode.values())
        print(f"Now total episodes: {self.num_episodes}, frames: {self.total_frames}")

    def write_meta(self, out_dir: Path):
        meta_dir = out_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        dataset_cursor = 0
        ep_rows_meta = []
        for (task_idx, _pq, _vid, ep_index) in sorted(self.episode_sources, key=lambda x: x[3]):
            n = self.lengths_by_episode.get(ep_index, 0)
            if n <= 0:
                continue
            ep_rows_meta.append(
                {
                    "episode_index": ep_index,
                    "tasks": [task_idx],
                    "length": n,
                    "dataset_from_index": dataset_cursor,
                    "dataset_to_index": dataset_cursor + (n - 1),
                    "robot_type": "g1",
                    "instruction": self.tasks_meta.get(task_idx, ""),
                }
            )
            dataset_cursor += n
        episodes_df = pd.DataFrame(ep_rows_meta).sort_values("episode_index").reset_index(drop=True)

        task_rows = [
            {"task_index": ti, "task": desc, "category": "default", "description": desc}
            for ti, desc in sorted(self.tasks_meta.items())
        ]
        tasks_df = pd.DataFrame(task_rows).sort_values("task_index").reset_index(drop=True)

        video_info = {
            "video.fps": float(self.fps), "video.codec": "h264", "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False, "has_audio": False,
        }
        features_meta = {
            "observation.images.egocentric": {
                "dtype": "video", "shape": list(self.video_shape),
                "names": ["height", "width", "channel"], "video_info": video_info,
            },
            "states": {"dtype": "float32", "shape": [-1]},
            "action": {"dtype": "float32", "shape": [-1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        }

        info = InfoDict(
            codebase_version=CODE_VERSION,
            robot_type="g1",
            total_episodes=self.num_episodes,
            total_frames=self.total_frames,
            total_tasks=len(self.tasks_meta),
            total_videos=self.num_episodes,
            total_chunks=math.ceil(self.num_episodes / self.chunks_size),
            chunks_size=self.chunks_size,
            fps=int(self.fps),
            data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            features=features_meta,
        )
        (meta_dir / "info.json").write_text(json.dumps(asdict(info), indent=4))
        with open(meta_dir / "tasks.jsonl", "w") as f:
            for row in tasks_df.to_dict(orient="records"):
                json.dump(row, f)
                f.write("\n")
        with open(meta_dir / "episodes.jsonl", "w") as f:
            for row in episodes_df.to_dict(orient="records"):
                json.dump(row, f)
                f.write("\n")
        print(f"\nWrote meta and {self.num_episodes} episode(s) into: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True,
                        help="LocoManiLab-prepared LeRobot dataset, e.g. /path/to/datasets/<task>")
    parser.add_argument("--work-dir", type=str, required=True,
                        help="Psi0 data root, e.g. $PSI_HOME/data/sonic/lerobot")
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--chunks-size", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve() / args.repo_id

    src_info = json.loads((data_root / "meta" / "info.json").read_text())
    fps = float(src_info["fps"])
    video_shape = src_info["features"][SRC_VIDEO_KEY]["shape"]
    print(f"Source: {src_info['total_episodes']} ep, fps={fps}, video={video_shape}")

    for d in [work_dir / "data", work_dir / "videos", work_dir / "meta"]:
        d.mkdir(parents=True, exist_ok=True)

    pipeline = LocoManiLabHeft2PsiConverter(fps=fps, video_shape=video_shape)
    pipeline.run(data_root, work_dir, args.chunks_size, args.num_workers)
    pipeline.write_meta(work_dir)


# Compatibility for code importing the previous converter class.
SomaSonic2PsiConverter = LocoManiLabHeft2PsiConverter

if __name__ == "__main__":
    main()
