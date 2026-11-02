"""Drop-in PolicyServer compatible with Isaac-GR00T's ZMQ REQ/REP protocol, backed by
a fine-tuned Psi0 checkpoint (SONIC / G1 78-D motion_token action space).

Why this file exists
--------------------
Psi0 ships its own RTC websocket server (psi_serve_rtc_token-sonic.py, FastAPI :8014)
for the real-robot RTC client. But LocoManiLab's whole-body deployment stack drives BOTH the
real G1 robot AND the tele_humanoid_benchmark sim through ONE contract: a
gr00t-compatible policy server on ``tcp://0.0.0.0:5550`` answering ``get_action``
(Isaac-GR00T PolicyClient, ZMQ REQ/REP + msgpack-numpy). gr00t serves it natively;
fastwam, lingbot-va, starVLA and PRTS each wrap their model into it. This file is the
Psi0 equivalent: plain chunked ``predict_action`` (no RTC), exactly like the other
baselines evaluate.

What it does per get_action()
------------------------------
  1. obs['video']['ego_view'] -> PIL -> the checkpoint's resize/center_crop transform.
  2. obs['state'][left_leg|right_leg|waist|left_arm|left_hand|right_arm|right_hand]
     (dataset 43-D layout) -> reorder to Psi0 states[43] = qpos(29)+hands(14)
     -> ActionStateTransform.normalize_state_func -> (1, 1, 43).
  3. model.predict_action(...) -> normalized (H, 78) -> maxmin.denormalize -> real.
  4. Split 78-D into parity keys:
        action.motion_token      [H, 64]
        action.left_hand_joints  [H, 7]
        action.right_hand_joints [H, 7]

Usage (inside the Psi0 uv env, cwd = Psi0 repo root):
    uv run --no-sync python src/psi/deploy/psi0_gr00t_server.py \
        --run-dir /path/to/outputs/psi0/sonic/<exp>.<ts> \
        --ckpt-step latest --host 0.0.0.0 --port 5550 --initial-prompt "..."

Wire protocol (matches gr00t.policy.server_client.PolicyServer):
    Transport : ZMQ REP socket on tcp://<host>:<port>
    Encoding  : msgpack-numpy, request = {"endpoint": str, "data": {...}}
    Endpoints : ping, kill, reset, get_action, get_modality_config
Dependencies beyond the psi group: pyzmq, msgpack-numpy.
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()  # PSI_HOME etc.; unlike train.py, missing .env is fine if paths are absolute

import argparse
import os
import os.path as osp
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from psi.config.config import LaunchConfig
from psi.models.psi0 import Psi0Model
from psi.utils import parse_args_to_tyro_config, pad_to_len, seed_everything

# ── 43-D G1 proprio layout (dataset meta/modality.json order) ────────────────
PROPRIO_DIM = 43
STATE_GROUPS = [
    ("left_leg", slice(0, 6)),
    ("right_leg", slice(6, 12)),
    ("waist", slice(12, 15)),
    ("left_arm", slice(15, 22)),
    ("left_hand", slice(22, 29)),
    ("right_arm", slice(29, 36)),
    ("right_hand", slice(36, 43)),
]
# dataset layout -> Psi0 states[43] = qpos(29: lower15+larm7+rarm7) + hands(14: lhand7+rhand7)
PSI_STATE_ORDER = [slice(0, 15), slice(15, 22), slice(29, 36), slice(22, 29), slice(36, 43)]
MOTION_TOKEN_DIM = 64
HAND_DIM = 7
ACTION_DIM = 78  # motion_token(64) + left_hand(7) + right_hand(7)

# ── controller variants ──────────────────────────────────────────────────────
# lab collects with three whole-body controllers and each one records a different
# action AND state space, so one checkpoint's output vector means nothing under
# another's controller. The checkpoint itself says which it is (its trained
# action_dim/odim), and the host's required action keys come from
# g1_backend/sim2real/vla/adapters.py:
#
#   sonic  78/43  motion_token   action.motion_token + left/right_hand_joints
#   heft   50/29  ref_qpos       action.ref_root_pos + ref_root_quat + ref_joint_pos (+hands)
#   amo    36/32  velocity_cmd   action.torso_vx/vy/vyaw (+target_yaw, rpy, height, arms, hands)
#
# State: everything except rpy/height is a slice of the host's 43-D joint vector; amo's
# torso pose is not, so the host sends it as state.rpy / state.height (LocoManiLab's sim eval
# client fills those in; a real-robot host must do the same).
VARIANTS = {
    78: {
        "name": "sonic",
        "state_dim": 43,
        # (key, slice into the 43-D joint vector) -- rpy/height come from obs["state"]
        "state_from_q43": PSI_STATE_ORDER,
        "state_extra": [],
        "action_split": [
            ("action.motion_token", slice(0, 64)),
            ("action.left_hand_joints", slice(64, 71)),
            ("action.right_hand_joints", slice(71, 78)),
        ],
    },
    50: {
        "name": "heft",
        "state_dim": 29,
        "state_from_q43": [slice(0, 15), slice(15, 22), slice(29, 36)],  # lower15+larm+rarm
        "state_extra": [],
        "action_split": [
            ("action.ref_root_pos", slice(0, 3)),
            ("action.ref_root_quat", slice(3, 7)),
            ("action.ref_joint_pos", slice(7, 36)),
            ("action.left_hand_joints", slice(36, 43)),
            ("action.right_hand_joints", slice(43, 50)),
        ],
    },
    36: {
        "name": "amo",
        "state_dim": 32,
        # lhand + rhand + larm + rarm  (28) ... then rpy(3) + height(1) from state_extra
        "state_from_q43": [slice(22, 29), slice(36, 43), slice(15, 22), slice(29, 36)],
        "state_extra": [("rpy", 3), ("height", 1)],
        "action_split": [
            ("action.left_hand", slice(0, 7)),
            ("action.right_hand", slice(7, 14)),
            ("action.left_arm", slice(14, 21)),
            ("action.right_arm", slice(21, 28)),
            ("action.rpy", slice(28, 31)),
            ("action.height", slice(31, 32)),
            ("action.torso_vx", slice(32, 33)),
            ("action.torso_vy", slice(33, 34)),
            ("action.torso_vyaw", slice(34, 35)),
            ("action.target_yaw", slice(35, 36)),
        ],
    },
}


def resolve_ckpt_step(run_dir: Path, ckpt_step: str) -> int:
    if ckpt_step != "latest":
        return int(ckpt_step)
    ckpts = sorted(
        int(m.group(1))
        for p in (run_dir / "checkpoints").glob("ckpt_*")
        if (m := re.match(r"ckpt_(\d+)$", p.name))
    )
    if not ckpts:
        raise FileNotFoundError(f"no ckpt_* under {run_dir}/checkpoints")
    return ckpts[-1]


class Psi0GrootPolicy:
    def __init__(self, run_dir: Path, ckpt_step: str, device: str,
                 initial_prompt: Optional[str], num_inference_steps: int,
                 exec_horizon: Optional[int]):
        assert osp.exists(run_dir), f"run_dir {run_dir} does not exist!"
        step = resolve_ckpt_step(run_dir, ckpt_step)
        assert osp.exists(run_dir / "checkpoints" / f"ckpt_{step}"), f"ckpt_{step} missing!"
        assert osp.exists(run_dir / "run_config.json"), "run_config.json missing!"

        # exactly how psi_serve_rtc_token-sonic.py restores the training config
        config_: LaunchConfig = parse_args_to_tyro_config(run_dir / "argv.txt")  # type: ignore
        conf = (run_dir / "run_config.json").open("r").read()
        launch_config = config_.model_validate_json(conf)
        seed_everything(launch_config.seed or 42)

        self.device = torch.device(device)
        print(f"[psi0] loading ckpt_{step} from {run_dir} ...", flush=True)
        self.model = Psi0Model.from_pretrained(run_dir, step, launch_config, device=device)
        self.model.to(device)
        self.model.eval()

        self.maxmin = launch_config.data.transform.field          # ActionStateTransform
        self.model_transform = launch_config.data.transform.model  # Psi0ModelTransform
        self.launch_cfg = launch_config
        self.Da = launch_config.model.action_dim                   # 78 sonic / 50 heft / 36 amo
        # The checkpoint's own dims decide the controller: nothing else in the run dir
        # records it, and a wrong guess is a silent misfire (the host would happily
        # apply 36 amo numbers as if they were the head of a sonic token).
        self.variant = VARIANTS.get(self.Da)
        if self.variant is None:
            known = ", ".join(f"{dim}={spec['name']}" for dim, spec in VARIANTS.items())
            raise SystemExit(
                f"checkpoint has action_dim={self.Da}, which matches no known controller "
                f"({known}). Add it to VARIANTS if this is a new action space.")
        odim = launch_config.model.odim
        if odim != self.variant["state_dim"]:
            raise SystemExit(
                f"checkpoint is action_dim={self.Da} ({self.variant['name']}) but odim={odim}, "
                f"expected {self.variant['state_dim']} — mismatched training config.")
        print(f"[psi0] controller variant: {self.variant['name']} "
              f"(action {self.Da} / state {odim})", flush=True)
        self.num_inference_steps = num_inference_steps
        self.exec_horizon = exec_horizon
        self._current_prompt = initial_prompt

        from torchvision.transforms import v2
        self._img_t = v2.Compose([self.model_transform.resize(), self.model_transform.center_crop()])

    # ---- public endpoints ---------------------------------------------------
    def ping(self) -> dict:
        return {"status": "ok", "message": "psi0 gr00t-compat server is running"}

    def get_modality_config(self) -> dict:
        return {}

    def reset(self, options: dict | None = None) -> dict:
        prompt = (options or {}).get("prompt")
        if prompt is not None:
            self._current_prompt = prompt
        return {"status": "reset", "prompt": self._current_prompt}

    def get_action(self, observation: dict[str, Any],
                   options: dict[str, Any] | None = None) -> tuple[dict, dict]:
        t0 = time.monotonic()
        prompt = self._extract_prompt(observation) or self._current_prompt
        if prompt is None:
            raise RuntimeError(
                "No prompt set. Pass --initial-prompt, call reset(prompt=...), or include "
                "obs['language']['annotation.human.task_description'].")
        self._current_prompt = prompt

        frame = self._extract_image(observation)              # [H,W,3] uint8
        img = self._img_t(Image.fromarray(frame))
        batch_images = [[img]]                                # batch of history-lists

        q43 = self._extract_proprio(observation)              # dataset layout, [43]
        parts = [q43[sl] for sl in self.variant["state_from_q43"]]
        for key, width in self.variant["state_extra"]:
            parts.append(self._extract_state_group(observation, key, width))
        psi_states = np.concatenate(parts).astype(np.float32)
        assert self.maxmin.normalize_state, "checkpoint was trained without state norm?"
        if self.maxmin.pad_state_dim is not None and self.maxmin.pad_state_dim != len(psi_states):
            psi_states = pad_to_len(psi_states, self.maxmin.pad_state_dim, dim=0)[0]
        obs = self.maxmin.normalize_state_func(psi_states)
        obs = np.asarray(obs, dtype=np.float32)[np.newaxis, np.newaxis, :]  # (1, 1, 43)

        t_inf = time.monotonic()
        with torch.no_grad():
            norm_actions = self.model.predict_action(
                observations=batch_images,
                states=torch.from_numpy(obs).to(self.device),
                traj2ds=None,
                instructions=[prompt.lower()],
                num_inference_steps=self.num_inference_steps,
            )[0].float().detach().cpu().numpy()               # (1, H, 78) -> (H, 78)
        infer_ms = (time.monotonic() - t_inf) * 1000.0

        raw = np.asarray(self.maxmin.denormalize(norm_actions), dtype=np.float32)  # (H, 78)
        if raw.ndim == 3:
            raw = raw[0]
        if self.exec_horizon is not None:
            raw = raw[: self.exec_horizon]

        action_dict = {
            key: np.ascontiguousarray(raw[:, sl], dtype=np.float32)
            for key, sl in self.variant["action_split"]
        }
        print(f"[psi0] get_action: H={raw.shape[0]} infer={infer_ms:.0f} ms prompt={prompt!r}",
              flush=True)
        info = {"inference_time_s": time.monotonic() - t0, "chunk_horizon": int(raw.shape[0])}
        return action_dict, info

    # ---- internals ----------------------------------------------------------
    @staticmethod
    def _extract_prompt(obs: dict) -> Optional[str]:
        lang = obs.get("language") or {}
        raw = lang.get("annotation.human.task_description")
        if raw is None:
            return None
        while isinstance(raw, (list, tuple)) and len(raw) > 0:
            if isinstance(raw[0], (list, tuple)):
                raw = raw[0]
            else:
                break
        if isinstance(raw, (list, tuple)):
            raw = raw[0] if raw else None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        return raw if isinstance(raw, str) else None

    @staticmethod
    def _extract_image(obs: dict) -> np.ndarray:
        video = obs.get("video") or {}
        img = None
        for k in ("ego_view", "observation.images.ego_view"):
            if k in video:
                img = video[k]
                break
        if img is None:
            if len(video) == 1:
                img = next(iter(video.values()))
            else:
                raise KeyError(f"No ego_view in obs['video']; keys={list(video.keys())}")
        img = np.asarray(img)
        while img.ndim > 3 and img.shape[0] >= 1:  # peel [B,T,..] to the newest frame
            img = img[-1]
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"ego_view frame must be [H,W,3], got {img.shape}")
        return img.astype(np.uint8)

    @staticmethod
    def _extract_state_group(obs: dict, key: str, width: int) -> np.ndarray:
        """A `state.<key>` entry that is NOT part of the 43-D joint vector.

        amo's state ends in the torso pose (rpy, height), which no joint slice can
        supply — the host has to measure and send it. Erroring here beats padding with
        zeros: a policy that trained on a moving torso pose and is fed a constant one
        looks like a bad checkpoint, not a missing observation.
        """
        state = obs.get("state") or {}
        if key not in state:
            raise KeyError(
                f"observation is missing state.{key} ({width}-D), which this checkpoint's "
                f"state vector needs. LocoManiLab's sim eval client sends it (torso_state() in "
                f"locomanilab/runtime/sim_client.py); a real-robot host must too.")
        v = np.asarray(state[key], dtype=np.float32).reshape(-1)
        if v.shape[0] != width:
            raise ValueError(f"state.{key} must be {width}-D, got {v.shape[0]}")
        return v

    @staticmethod
    def _extract_proprio(obs: dict) -> np.ndarray:
        q = obs.get("q")
        if q is not None:
            q = np.asarray(q, dtype=np.float32).reshape(-1)
            if q.shape[0] != PROPRIO_DIM:
                raise ValueError(f"obs['q'] must be {PROPRIO_DIM}-dim, got {q.shape[0]}")
            return q
        state = obs.get("state") or {}
        order = [g for g, _ in STATE_GROUPS]
        missing = [g for g in order if g not in state]
        if missing:
            raise KeyError(f"obs['state'] missing groups {missing}; got {list(state.keys())}")
        parts = []
        for g in order:
            v = np.asarray(state[g], dtype=np.float32)
            while v.ndim > 1:   # peel [B,T,..] history to the newest frame
                v = v[-1]
            parts.append(v.reshape(-1))
        q = np.concatenate(parts)
        if q.shape[0] != PROPRIO_DIM:
            raise ValueError(f"Reconstructed proprio is {q.shape[0]}-dim, expected {PROPRIO_DIM}")
        return q


# ── ZMQ server (mirrors gr00t.policy.server_client.PolicyServer wire format) ─
def serve(policy: Psi0GrootPolicy, host: str, port: int) -> None:
    import msgpack_numpy as mnp
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[psi0] PolicyServer ready on tcp://{host}:{port}", flush=True)

    running = True
    while running:
        try:
            req = mnp.unpackb(sock.recv(), raw=False)
            endpoint = req.get("endpoint", "get_action")
            data = req.get("data") or {}

            if endpoint == "ping":
                resp = policy.ping()
            elif endpoint == "kill":
                running = False
                resp = {"status": "killing"}
            elif endpoint == "reset":
                resp = policy.reset(options=data.get("options"))
            elif endpoint == "get_modality_config":
                resp = policy.get_modality_config()
            elif endpoint == "get_action":
                action, info = policy.get_action(
                    observation=data.get("observation", {}),
                    options=data.get("options"),
                )
                resp = (action, info)
            else:
                resp = {"error": f"unknown endpoint {endpoint!r}"}
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            resp = {"error": f"{type(e).__name__}: {e}"}
        try:
            sock.send(mnp.packb(resp, use_bin_type=True))
        except Exception:
            traceback.print_exc()
            break
    print("[psi0] server shut down", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Psi0 run dir (has run_config.json / argv.txt / checkpoints/)")
    ap.add_argument("--ckpt-step", type=str, default="latest")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5550)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--initial-prompt", type=str, default=None)
    ap.add_argument("--num-inference-steps", type=int, default=8)
    ap.add_argument("--exec-horizon", type=int, default=None,
                    help="truncate the returned chunk (default: full action_chunk_size)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    policy = Psi0GrootPolicy(
        run_dir=args.run_dir.expanduser().resolve(),
        ckpt_step=args.ckpt_step,
        device=args.device,
        initial_prompt=args.initial_prompt,
        num_inference_steps=args.num_inference_steps,
        exec_horizon=args.exec_horizon,
    )
    serve(policy, args.host, args.port)


if __name__ == "__main__":
    main()
