import numpy as np
import torch as th
from torch.nn import functional as F
 
from stable_baselines3.sac.sac import SAC
from stable_baselines3.common.utils import polyak_update
 
 
class SmoothSAC(SAC):
    def __init__(
        self,
        *args,
        smooth_coef: float = 0.01,
        smooth_lipschitz: float = 0.2,
        smooth_noise_std: float = 0.02,
        temporal_coef: float = 0.1,
        max_grad_norm: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.smooth_coef = smooth_coef
        self.smooth_lipschitz = smooth_lipschitz
        self.smooth_noise_std = smooth_noise_std
        self.temporal_coef = temporal_coef
        self.presence_mean = 0.0
        self.gap_mean = 0.0
        self.max_grad_norm = max_grad_norm
        
 
    def _smoothness_penalty(self, replay_data, actions_pi: th.Tensor) -> th.Tensor:
        obs = replay_data.observations
        agents = obs["agents"]
        batch_size = agents.shape[0]

        presence = (agents[..., 7] != 0).float().unsqueeze(-1)
        #presence = (agents[..., 5] != 0).float().unsqueeze(-1)
        self.presence_mean = presence.mean().item()

        scale = th.empty(batch_size, 1, 1, device=agents.device).uniform_(0.5, 5.0)
        noise = th.randn_like(agents) * self.smooth_noise_std * scale
        noise[..., 7] = 0.0
        noise = noise * presence

        perturbed_obs = dict(obs)
        perturbed_obs["agents"] = agents + noise

        # deterministic=True -> tanh(mean_actions), bypasses rsample() entirely
        actions_pi_det = self.actor(obs, deterministic=True)
        actions_pi_nb  = self.actor(perturbed_obs, deterministic=True)

        d = noise.flatten(1).norm(dim=1, keepdim=True) + 1e-8
        gap = (actions_pi_det - actions_pi_nb).norm(dim=1, keepdim=True)
        self.gap_mean = gap.mean().item()

        d_cap = 3.0
        allowed_gap = self.smooth_lipschitz * d_cap * th.tanh(d/d_cap)
        mean_actions, log_std, _ = self.actor.get_action_dist_params(obs)
        self.sat_frac = (mean_actions.abs() > 2.5).float().mean().item()   # tanh(2.5)≈0.99
        self.active_frac = (gap > allowed_gap).float().mean().item()

        return th.relu(gap - allowed_gap).mean()
 
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # --- everything below is copied verbatim from SAC.train() (sb3 v2.9.0) ---
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
 
        self._update_learning_rate(optimizers)
 
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        smooth_penalties = []  # NEW: for logging
        actor_grad_norms = []  # NEW: for logging
 
        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
 
            if self.use_sde:
                self.actor.reset_noise()
 
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)
 
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
 
            ent_coefs.append(ent_coef.item())
 
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
 
            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
 
            current_q_values = self.critic(replay_data.observations, replay_data.actions)
 
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]
 
            self.critic.optimizer.zero_grad()
            critic_loss.backward()

            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                critic_grad_norm = th.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), 
                    max_norm=self.max_grad_norm
                )

            self.critic.optimizer.step()
 
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
 
            # ------------------- L2C2 SMOOTHNESS TERM (NEW) -------------------
            smooth_penalty = self._smoothness_penalty(replay_data, actions_pi)
            actor_loss = actor_loss + self.smooth_coef * smooth_penalty
            smooth_penalties.append(smooth_penalty.item())
            # --------------------------------------------------------------------

            a_t   = self.actor(replay_data.observations, deterministic=True)
            a_tp1 = self.actor(replay_data.next_observations, deterministic=True)
            temporal = (a_t - a_tp1).pow(2).sum(dim=1).mean()
            actor_loss = actor_loss + self.temporal_coef * temporal 

            actor_losses.append(actor_loss.item())
 
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            actor_gn = th.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=float("inf"))
            actor_grad_norms.append(actor_gn.item())
            self.actor.optimizer.step()
 
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
 
        self._n_updates += gradient_steps
 
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/smooth_penalty", np.mean(smooth_penalties))  # NEW
        self.logger.record("train/presence_mean", np.mean(self.presence_mean))
        self.logger.record("train/gap_mean", np.mean(self.gap_mean))
        self.logger.record("train/sat_frac", self.sat_frac)
        self.logger.record("train/active_frac", self.active_frac)
        self.logger.record("train/actor_grad_norm", np.mean(actor_grad_norms))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        