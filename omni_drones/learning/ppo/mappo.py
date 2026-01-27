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
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from torchrl.data import Composite, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
import logging

from ..utils.valuenorm import ValueNorm1
from ..modules.distributions import IndependentNormal
from .common import GAE

@dataclass
class PPOConfig:
    name: str = "mappo"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16

    share_actor: bool = True

cs = ConfigStore.instance()
cs.store("mappo", node=PPOConfig, group="algo")


def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)
        scale = torch.exp(self.actor_std).expand_as(loc)
        return loc, scale


class MAPPOPolicy(TensorDictModuleBase):

    def __init__(
        self,
        cfg: PPOConfig,
        observation_spec: Composite,
        action_spec: Composite,
        reward_spec: TensorSpec,
        device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.entropy_coef = 0.001
        self.clip_param = 0.1
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.n_agents, self.action_dim = action_spec[("agents", "action")].shape[-2:]
        self.gae = GAE(0.99, 0.95)

        fake_input = observation_spec.zero()

        if cfg.share_actor:
            actor_module=TensorDictModule(
                nn.Sequential(make_mlp([256, 256, 256]), Actor(self.action_dim)),
                [("agents", "observation")], ["loc", "scale"]
            )
        else:
            raise NotImplementedError("Non-shared actor not implemented for MAPPOPolicy")
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True,
            log_prob_key="sample_log_prob"
        ).to(self.device)

        if ("agents", "observation_central") in observation_spec.keys(True):
            critic_input = [("agents", "observation_central")]
            # centralized = True
        else:
            logging.warning("No central observation found, using local observation for critic.")
            critic_input = [("agents", "observation")]
            # centralized = False

        # critic has to output reward for each agent
        self.critic = TensorDictModule(
            nn.Sequential(make_mlp([256, 256, 256]), nn.LazyLinear(reward_spec['agents']['reward'].shape[-2])),
            critic_input, ["state_value"]
        ).to(self.device)

        self.actor(fake_input)
        self.critic(fake_input)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)

        self.actor.apply(init_)
        self.critic.apply(init_)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=5e-4)
        reward_shape = reward_spec[("agents", "reward")].shape[-2:]
        # if centralized:
            # reward_shape = torch.Size([1, reward_spec[("agents", "reward")].shape[-1]])
        self.value_norm = ValueNorm1(reward_shape).to(self.device)

    def __call__(self, tensordict: TensorDict):
        self.actor(tensordict)
        self.critic(tensordict)
        tensordict.exclude("loc", "scale", "feature", inplace=True)
        return tensordict

    def train_op(self, tensordict: TensorDict):
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_values = self.critic(next_tensordict)["state_value"]
        rewards = tensordict[("next", "agents", "reward")]
        dones = (
            tensordict[("next", "done")]
            .expand(-1, -1, self.n_agents)
            .unsqueeze(-1)
        )
        values = tensordict["state_value"]
        if len(values.shape) == 3:
            values = values.unsqueeze(-1)
        if len(next_values.shape) == 3:
            next_values = next_values.unsqueeze(-1)
        num_agents = tensordict["agents"]["action"].shape[-2]
        if values.shape[-2] == 1:
            values = values.repeat_interleave(num_agents, dim=-2)
        if next_values.shape[-2] == 1:
            next_values = next_values.repeat_interleave(num_agents, dim=-2)
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        infos = []
        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))

        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}

    def _update(self, tensordict: TensorDict):
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[("agents", "action")])
        entropy = dist.entropy()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2))
        entropy_loss = - self.entropy_coef * torch.mean(entropy)

        b_values = tensordict["state_value"]
        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        values_clipped = b_values + (values - b_values).clamp(
            -self.clip_param, self.clip_param
        )
        if len(values_clipped.shape) == 2:
            values_clipped = values_clipped.unsqueeze(-1)
        if len(values.shape) == 2:
            values = values.unsqueeze(-1)
        value_loss_clipped = self.critic_loss_fn(b_returns, values_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)
        value_loss = torch.max(value_loss_original, value_loss_clipped)

        # Update actor
        actor_loss = policy_loss + entropy_loss
        self.actor_opt.zero_grad()
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 5)
        self.actor_opt.step()

        # Update critic
        self.critic_opt.zero_grad()
        value_loss.backward()
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 5)
        self.critic_opt.step()

        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        return TensorDict({
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var
        }, [])


def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1)
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]
