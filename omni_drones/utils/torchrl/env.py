# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import numpy as np
import einops
from tqdm import tqdm
from typing import Optional, Sequence
import warnings

from dataclasses import dataclass
from torchrl.envs import EnvBase
from torchrl.data import TensorSpec, Composite
from tensordict import TensorDictBase


@dataclass
class AgentSpec:
    name: str
    n: int
    observation_key: Optional[str] = "observation"
    action_key: Optional[str] = None
    state_key: Optional[str] = None
    reward_key: Optional[str] = None
    done_key: Optional[str] = None

    _env: Optional[EnvBase] = None

    @property
    def observation_spec(self) -> TensorSpec:
        return self._env.observation_spec[self.observation_key]

    @property
    def action_spec(self) -> TensorSpec:
        if self.action_key is None:
            return self._env.action_spec
        try:
            return self._env.input_spec["full_action_spec"][self.action_key]
        except KeyError:
            return self._env.action_spec[self.action_key]

    @property
    def state_spec(self) -> TensorSpec:
        if self.state_key is None:
            raise ValueError()
        return self._env.observation_spec[self.state_key]

    @property
    def reward_spec(self) -> TensorSpec:
        if self.reward_key is None:
            return self._env.reward_spec
        try:
            return self._env.output_spec["full_reward_spec"][self.reward_key]
        except KeyError:
            return self._env.reward_spec[self.reward_key]

    @property
    def done_spec(self) -> TensorSpec:
        if self.done_key is None:
            return self._env.done_spec
        try:
            return self._env.output_spec["full_done_spec"][self.done_key]
        except KeyError:
            return self._env.done_spec[self.done_key]


class RenderCallback:

    def __init__(self, interval: int=2):
        self.interval = interval
        self.frames = []
        self.i = 0
        self.t = tqdm(desc="Rendering")
        self._viewport_warning_shown = False

    def __call__(self, env, *args):
        if self.i % self.interval == 0:
            # Check if viewport is enabled before attempting to render
            # The enable_viewport attribute is on the base environment (IsaacEnv)
            # Check both the env itself and its base_env if it exists
            base_env = getattr(env, 'base_env', env)
            enable_viewport = getattr(base_env, 'enable_viewport', None)
            
            # Check if Replicator is enabled (required for RGB rendering)
            enable_replicator = False
            if hasattr(base_env, 'cfg') and hasattr(base_env.cfg, 'sim'):
                enable_replicator = getattr(base_env.cfg.sim, 'enable_replicator', False)
            
            if enable_viewport is False:
                if not self._viewport_warning_shown:
                    warnings.warn(
                        "RenderCallback: Cannot render 'rgb_array' when viewport is disabled. "
                        "Skipping rendering. To enable rendering, initialize the environment with "
                        "headless=False or set enable_viewport=True.",
                        UserWarning
                    )
                    self._viewport_warning_shown = True
            elif not enable_replicator:
                if not self._viewport_warning_shown:
                    warnings.warn(
                        "RenderCallback: Cannot render 'rgb_array' when Replicator is disabled. "
                        "Skipping rendering. To enable rendering, set cfg.sim.enable_replicator=True.",
                        UserWarning
                    )
                    self._viewport_warning_shown = True
            else:
                try:
                    frame = env.render(mode="rgb_array")
                    self.frames.append(frame)
                    self.t.update(self.interval)
                except RuntimeError as e:
                    error_msg = str(e)
                    if "enable viewport is False" in error_msg or "RGB rendering requires Replicator" in error_msg:
                        if not self._viewport_warning_shown:
                            warnings.warn(
                                f"RenderCallback: {error_msg}. Skipping rendering.",
                                UserWarning
                            )
                            self._viewport_warning_shown = True
                    else:
                        raise
        self.i += 1
        return self.i

    def get_video_array(self, axes: str = "t c h w"):
        if len(self.frames) == 0:
            return None
        return einops.rearrange(np.stack(self.frames), "t h w c -> " + axes)


class EpisodeStats:
    def __init__(self, in_keys: Sequence[str] = None):
        self.in_keys = in_keys
        self._stats = []
        self._episodes = 0

    def add(self, tensordict: TensorDictBase) -> TensorDictBase:
        next_tensordict = tensordict["next"]
        done = next_tensordict.get("done")
        if done.any():
            done = done.squeeze(-1)
            self._episodes += done.sum().item()
            next_tensordict = next_tensordict.select(*self.in_keys)
            self._stats.extend(next_tensordict[done].cpu().unbind(0))
        return len(self)

    def pop(self):
        stats: TensorDictBase = torch.stack(self._stats).to_tensordict()
        self._stats.clear()
        return stats

    def __len__(self):
        return len(self._stats)

