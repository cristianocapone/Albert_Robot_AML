"""
robot_ppo_tutorial.py
─────────────────────────────────────────────────────────────────────────────
Teaching script: train a 4-legged robot to walk using PPO.

WHAT THIS SCRIPT DOES
  1. Simulates a quadruped robot in MuJoCo.
  2. Trains an actor-critic neural network with Proximal Policy Optimization.
  3. Saves the best checkpoint and records a short video.

CORE CONCEPTS COVERED
  • Markov Decision Process  (state, action, reward, next-state)
  • Actor-Critic architecture (shared perception, split outputs)
  • PPO: clipped surrogate objective, value function loss, entropy bonus
  • Generalized Advantage Estimation (GAE)
  • ΔΔθ ("delta-delta-theta") acceleration-level joint control

DEPENDENCIES
  pip install torch mujoco mediapy matplotlib tqdm

HOW TO RUN
  python robot_ppo_tutorial.py

OUTPUT
  models/best.pth          — saved checkpoint
  movies/rollout_NNN.mp4   — rendered video every RENDER_EVERY episodes
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Imports
# ─────────────────────────────────────────────────────────────────────────────
import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mujoco
import mediapy as media
import matplotlib.pyplot as plt
from tqdm import trange

os.makedirs("models", exist_ok=True)
os.makedirs("movies", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — MuJoCo environment
# ─────────────────────────────────────────────────────────────────────────────

# Load the robot description (XML defines geometry, joints, actuators).
mjmodel = mujoco.MjModel.from_xml_path("dog.xml")
mjdata  = mujoco.MjData(mjmodel)

# SUBSTEPS: how many physics steps per control step.
# More substeps = more accurate physics, but slower training.
SUBSTEPS = 5
CTRL_DT  = mjmodel.opt.timestep * SUBSTEPS   # wall-clock duration of one action

EPISODE_DUR = 8.0    # seconds per training episode
SETTLE_TIME = 0.3    # seconds to let the robot settle into standing pose at reset

print(f"Control rate : {1/CTRL_DT:.0f} Hz   (dt = {CTRL_DT*1000:.1f} ms)")
print(f"Steps/episode: {int(EPISODE_DUR / CTRL_DT)}")

# ── Joint geometry ────────────────────────────────────────────────────────────
# Each leg has 2 joints: hip and knee.  4 legs → 8 joints total.
# Joint order: FL-hip, FL-knee, FR-hip, FR-knee, BL-hip, BL-knee, BR-hip, BR-knee

HIP_INIT  =  0.90   # rad — standing pose for hip
KNEE_INIT = -1.40   # rad — standing pose for knee

# Safety limits: the joint cannot leave these ranges.
HIP_LO,  HIP_HI  = HIP_INIT  - 0.5, HIP_INIT  + 0.5
KNEE_LO, KNEE_HI = KNEE_INIT - 0.3, KNEE_INIT + 0.3

# Replicate the hip/knee limits across all 4 legs.
OFFSETS  = np.array([HIP_INIT,  KNEE_INIT]  * 4, dtype=np.float32)
JOINT_LO = np.array([HIP_LO,   KNEE_LO]    * 4, dtype=np.float32)
JOINT_HI = np.array([HIP_HI,   KNEE_HI]    * 4, dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — State, action, and control scheme
# ─────────────────────────────────────────────────────────────────────────────

STATE_DIM  = 24   # jpos(8) + jvel(8) + delta(8)  — what the policy "sees"
ACTION_DIM = 8    # one ΔΔθ value per joint
HIDDEN_DIM = 64   # hidden units in the actor (small enough for microcontrollers)

# ── ΔΔθ control (acceleration-level) ─────────────────────────────────────────
#
# Naive approach: action = absolute joint angle  (position-level)
#   Problem: can produce jerky, discontinuous motion.
#
# Better approach: action = change to joint angle  (velocity-level)
#   Still can produce large, sudden velocity changes.
#
# This script: action = change to the VELOCITY (= acceleration-level)
#   The network outputs ΔΔθ — a small nudge to a persistent "delta buffer".
#   The delta buffer accumulates like velocity; joint target accumulates like position.
#
#   delta_new = clip( delta + action * MAX_DDELTA,  −DELTA_LIMIT, +DELTA_LIMIT )
#   ctrl_new  = clip( ctrl  + delta_new,             JOINT_LO,     JOINT_HI    )
#
# Effect: the robot cannot instantly reverse direction; gait smoothness comes
# for free without any explicit frequency or smoothness penalty.

MAX_DDELTA   = 0.02   # how much one action can change the delta buffer (rad/step)
DELTA_LIMIT  = 0.05   # maximum absolute delta (caps the "velocity", rad/step)


def apply_action(ctrl, delta, action):
    """Apply one ΔΔθ action; returns (new_ctrl, new_delta)."""
    delta_new = np.clip(delta + action * MAX_DDELTA, -DELTA_LIMIT, DELTA_LIMIT)
    ctrl_new  = np.clip(ctrl  + delta_new,            JOINT_LO,    JOINT_HI)
    return ctrl_new, delta_new


def get_state(delta):
    """Build the 24-dim observation vector from simulator + delta buffer.

    The delta buffer is part of the state so the policy knows its own "momentum"
    and can plan whether to accelerate or brake.
    """
    jpos  = mjdata.qpos[7:15].astype(np.float32)   # 8 joint angles (rad)
    jvel  = mjdata.qvel[6:14].astype(np.float32)   # 8 joint velocities (rad/s)
    return np.concatenate([jpos, jvel, delta])


def orientation():
    """Return (roll, pitch, yaw) in radians from the robot's quaternion."""
    qw, qx, qy, qz = mjdata.qpos[3:7]
    roll  = math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
    pitch = math.asin(float(np.clip(2*(qw*qy - qz*qx), -1, 1)))
    yaw   = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
    return roll, pitch, yaw


def is_fallen():
    """Return True if the robot has tipped over or crashed to the ground."""
    _, pitch, _ = orientation()
    return mjdata.qpos[2] < 0.03 or abs(pitch) > math.radians(45)


def reset():
    """Reset the simulation; return (ctrl, delta) both at the neutral pose."""
    neutral = np.array([HIP_INIT, KNEE_INIT] * 4, dtype=np.float32)
    mujoco.mj_resetData(mjmodel, mjdata)
    mjdata.qpos[7:15] = neutral
    mjdata.qpos[2]    = 0.10    # lift robot 10 cm off the ground at reset
    mujoco.mj_forward(mjmodel, mjdata)
    # Run physics for SETTLE_TIME with neutral pose so the robot is stable.
    for _ in range(int(SETTLE_TIME / mjmodel.opt.timestep)):
        mjdata.ctrl[:] = neutral
        mujoco.mj_step(mjmodel, mjdata)
    return neutral.copy(), np.zeros(ACTION_DIM, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — Reward function
# ─────────────────────────────────────────────────────────────────────────────

# The reward tells the robot WHAT we want.  All reward weights are here so
# you can tweak them in one place and see what changes.

TARGET_VX    = 0.125   # desired forward speed (m/s) — 1 m in 8 s
VX_TRACK_W   = 4.0     # how hard to penalize speed mismatch
VX_TOL       = 0.02    # dead-zone: no penalty if speed is within ±VX_TOL of target

ALIVE_BONUS  = 0.05    # small constant reward just for staying upright
SIDE_VEL_W   = 0.10    # penalize sideways drift
ACTION_REG_W = 0.0001  # penalize large accelerations (keeps motion smooth)
DELTA_REG_W  = 0.0002  # penalize large deltas (discourages drifting far from neutral)
JVEL_W       = 0.05    # reward active joint movement (encourages locomotion)
JEXC_W       = 0.05    # reward joint excursion from neutral (encourages big steps)
YAW_W        = 0.20    # penalize turning left/right (stay straight)
FALL_PENALTY = 5.0     # one-time penalty on falling


def compute_reward(action, delta, yaw0):
    """Compute scalar reward for the current simulator state."""
    vx   = float(mjdata.qvel[0])       # forward speed
    vy   = abs(float(mjdata.qvel[1]))  # sideways speed (always positive)
    jvel = float(np.mean(np.abs(mjdata.qvel[6:14])))              # mean joint velocity
    jpos = mjdata.qpos[7:15].astype(np.float32)
    excursion = float(np.mean(np.abs(jpos - OFFSETS)))            # avg deviation from neutral

    _, _, yaw = orientation()
    dyaw = (yaw - yaw0 + math.pi) % (2 * math.pi) - math.pi     # signed yaw change

    # Velocity-tracking term: zero at TARGET_VX, falls as squared error.
    # The dead-zone ignores tiny fluctuations that are physically unavoidable.
    vx_err = max(0.0, abs(vx - TARGET_VX) - VX_TOL)
    r_vel  = -VX_TRACK_W * vx_err ** 2

    return (r_vel
            + ALIVE_BONUS
            - SIDE_VEL_W   * vy
            - ACTION_REG_W * float(np.sum(action ** 2))
            - DELTA_REG_W  * float(np.sum(delta  ** 2))
            + JVEL_W       * jvel
            + JEXC_W       * excursion
            - YAW_W        * dyaw ** 2)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — Actor-Critic neural network
# ─────────────────────────────────────────────────────────────────────────────

# PPO uses one network with two "heads":
#   Actor (π): maps state → action distribution  (what to do)
#   Critic (V): maps state → scalar value         (how good is this state)
#
# The critic is only used during training to estimate advantages.
# At deployment, only the actor runs on the robot.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# Gaussian policy parameters
LOG_STD_INIT = 0.5    # initial log-std — moderately exploratory
LOG_STD_MIN  = -2.0   # min log-std (clips std to e^-2 ≈ 0.14)
LOG_STD_MAX  =  1.0   # max log-std (clips std to e^1  ≈ 2.72)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()

        # ── Actor: 24 → 64 → 8  (small, runs on microcontroller) ─────────
        # We use a single hidden layer with ReLU.  The output layer is linear
        # (no activation); we apply tanh to the *sample*, not the mean.
        self.pi = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, ACTION_DIM),
        )
        # Small output weights → small initial actions → stable start
        nn.init.orthogonal_(self.pi[-1].weight, gain=0.01)
        nn.init.zeros_(self.pi[-1].bias)

        # Learnable log standard deviation (one value per action dimension).
        # Separate from the actor MLP so exploration decays independently.
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), LOG_STD_INIT))

        # ── Critic: 24 → 256 → 256 → 1  (larger; stays on the PC) ───────
        self.vf = nn.Sequential(
            nn.Linear(STATE_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
            nn.Linear(256, 1),
        )
        nn.init.orthogonal_(self.vf[-1].weight, gain=1.0)
        nn.init.zeros_(self.vf[-1].bias)

    # ── Helper: sample an action and compute its log-probability ──────────
    def get_action(self, s, deterministic=False):
        """
        s: (batch, STATE_DIM) float tensor

        Returns (action, log_prob, value).
          action:   tanh-squashed sample in (-1, 1)^ACTION_DIM
          log_prob: log π(action | s)  — needed by PPO
          value:    V(s)               — needed for GAE
        """
        mean    = self.pi(s)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std     = log_std.exp()

        # At evaluation we take the mean (no noise); during training we sample.
        raw = mean if deterministic else mean + std * torch.randn_like(std)

        # Squash with tanh so actions always lie in (-1, 1).
        action = torch.tanh(raw)

        # Log-prob of a tanh-Gaussian:
        #   log π = log N(raw | mean, std)  −  log(1 − tanh²(raw))
        # The second term corrects for the tanh change-of-variables.
        log_prob = (
            -0.5 * ((raw - mean) / (std + 1e-8)) ** 2
            - log_std
            - 0.5 * math.log(2 * math.pi)
            - torch.log(1 - action.pow(2) + 1e-6)
        ).sum(-1)

        value = self.vf(s).squeeze(-1)
        return action, log_prob, value

    # ── Helper: re-evaluate stored actions (used in PPO update) ──────────
    def evaluate_actions(self, s, actions_tanh):
        """
        Re-compute log_prob and entropy for stored (tanh-space) actions.
        This is called on mini-batches during the PPO update.
        """
        mean    = self.pi(s)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std     = log_std.exp()

        # Invert tanh to recover the raw (pre-squash) action.
        raw = torch.atanh(actions_tanh.clamp(-0.999, 0.999))

        log_prob = (
            -0.5 * ((raw - mean) / (std + 1e-8)) ** 2
            - log_std
            - 0.5 * math.log(2 * math.pi)
            - torch.log(1 - actions_tanh.pow(2) + 1e-6)
        ).sum(-1)

        # Differential entropy of a Gaussian (ignoring the tanh correction —
        # good enough for the entropy bonus term in the PPO loss).
        entropy = (log_std + 0.5 * math.log(2 * math.pi * math.e)).sum(-1)

        value = self.vf(s).squeeze(-1)
        return log_prob, entropy, value


ac  = ActorCritic().to(device)
opt = torch.optim.Adam(ac.parameters(), lr=3e-4, eps=1e-5)

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — Experience replay buffer + GAE
# ─────────────────────────────────────────────────────────────────────────────

# PPO is an ON-POLICY algorithm: it collects experience with the CURRENT policy,
# updates the network, then throws the data away.
# The buffer stores one "batch" of transitions before each update.

N_STEPS_PER_UPDATE = 4096   # collect this many steps, then do one PPO update
PPO_EPOCHS         = 10     # how many passes over the buffer per update
MINIBATCH_SIZE     = 256    # mini-batch size for SGD
GAMMA              = 0.99   # discount factor  (future rewards worth less)
GAE_LAMBDA         = 0.95   # GAE smoothing    (0 = TD(0), 1 = Monte Carlo)
CLIP_EPS           = 0.2    # PPO clipping range
ENTROPY_COEF       = 0.005  # entropy bonus weight (encourages exploration)
VF_COEF            = 0.5    # value-function loss weight
MAX_GRAD           = 0.5    # gradient clipping threshold


class RolloutBuffer:
    """Stores one batch of transitions collected from the environment."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.states, self.actions = [], []
        self.log_probs, self.rewards = [], []
        self.values, self.dones = [], []

    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)

    def compute_advantages(self, last_value):
        """Generalized Advantage Estimation (GAE, Schulman 2016).

        GAE interpolates between the high-variance Monte-Carlo return and
        the low-variance (but biased) 1-step TD estimate.
        λ = 0  →  pure 1-step TD (low variance, high bias)
        λ = 1  →  full Monte Carlo (high variance, low bias)

        The advantage A(s,a) = Q(s,a) − V(s) tells us whether action a was
        better or worse than what the critic expected.
        """
        T      = len(self.rewards)
        adv    = np.zeros(T, dtype=np.float32)
        values = np.array(self.values + [last_value], dtype=np.float32)
        rew    = np.array(self.rewards, dtype=np.float32)
        done   = np.array(self.dones,   dtype=np.float32)

        gae = 0.0
        for t in reversed(range(T)):
            # δ_t = r_t + γ V(s_{t+1}) − V(s_t)   (TD error)
            delta = rew[t] + GAMMA * values[t + 1] * (1.0 - done[t]) - values[t]
            # A_t = δ_t + γλ A_{t+1}   (recursive GAE)
            gae   = delta + GAMMA * GAE_LAMBDA * (1.0 - done[t]) * gae
            adv[t] = gae

        returns = adv + np.array(self.values, dtype=np.float32)
        return returns, adv

    def as_tensors(self, last_value):
        returns, adv = self.compute_advantages(last_value)

        S   = torch.FloatTensor(np.array(self.states)).to(device)
        A   = torch.FloatTensor(np.array(self.actions)).to(device)
        LP  = torch.FloatTensor(np.array(self.log_probs)).to(device)
        R   = torch.FloatTensor(returns).to(device)
        Adv = torch.FloatTensor(adv).to(device)
        # Normalize advantages: zero mean, unit variance within the batch.
        # This stabilizes learning by preventing very large or small gradients.
        Adv = (Adv - Adv.mean()) / (Adv.std() + 1e-8)
        return S, A, LP, R, Adv


buf = RolloutBuffer()

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 — PPO update
# ─────────────────────────────────────────────────────────────────────────────

def ppo_update(last_value):
    """Run PPO_EPOCHS passes over the buffer and update the network.

    PPO OBJECTIVE (actor loss):
      L = E[ min( r_t * A_t,  clip(r_t, 1−ε, 1+ε) * A_t ) ]
    where r_t = π_new(a|s) / π_old(a|s)  is the probability ratio.

    The clip keeps the new policy from moving too far from the old one in a
    single update, avoiding the catastrophic policy collapse that plagued
    earlier policy-gradient methods (like TRPO's much more expensive trust region).

    TOTAL LOSS:
      − actor_loss  +  VF_COEF * critic_loss  −  ENTROPY_COEF * entropy
    (we minimize, so actor_loss is negated because it's a maximization objective)
    """
    S, A, LP_old, R, Adv = buf.as_tensors(last_value)
    N = S.shape[0]

    for _ in range(PPO_EPOCHS):
        # Shuffle and iterate over mini-batches.
        for start in range(0, N, MINIBATCH_SIZE):
            idx = torch.randperm(N, device=device)[start:start + MINIBATCH_SIZE]

            lp_new, entropy, v_new = ac.evaluate_actions(S[idx], A[idx])

            # Probability ratio π_new / π_old  (in log-space for numerical stability)
            ratio = (lp_new - LP_old[idx]).exp()

            # Clipped surrogate objective
            surr1    = ratio * Adv[idx]
            surr2    = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * Adv[idx]
            pi_loss  = -torch.min(surr1, surr2).mean()

            # Critic regression loss
            vf_loss  = F.mse_loss(v_new, R[idx])

            loss = pi_loss + VF_COEF * vf_loss - ENTROPY_COEF * entropy.mean()

            opt.zero_grad()
            loss.backward()
            # Clip gradients to prevent exploding updates.
            nn.utils.clip_grad_norm_(ac.parameters(), MAX_GRAD)
            opt.step()

    buf.clear()

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 7 — Episode rollout
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(deterministic=False, render=False):
    """Run one episode; return (total_reward, distance_walked, frames, last_value).

    deterministic=True  → use actor mean (no noise)  — for evaluation
    deterministic=False → sample from Gaussian        — for training
    render=True         → capture frames for a video
    """
    ctrl, delta = reset()
    x0          = mjdata.qpos[0]   # starting x position (for measuring distance)
    _, _, yaw0  = orientation()
    total_r     = 0.0
    frames      = []
    renderer    = None

    if render:
        renderer = mujoco.Renderer(mjmodel, height=480, width=640)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(mjmodel, cam)
        cam.distance  = 0.8
        cam.elevation = -20

    s = get_state(delta)

    for _ in range(int(EPISODE_DUR / CTRL_DT)):
        # ── Actor inference ───────────────────────────────────────────────
        st = torch.FloatTensor(s).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob, value = ac.get_action(st, deterministic)
        a_np = action.squeeze(0).cpu().numpy()   # shape (8,), values in (−1, 1)

        # ── Step the physics ──────────────────────────────────────────────
        ctrl, delta = apply_action(ctrl, delta, a_np)
        mjdata.ctrl[:] = ctrl
        for _ in range(SUBSTEPS):
            mujoco.mj_step(mjmodel, mjdata)

        # ── Reward ────────────────────────────────────────────────────────
        r    = compute_reward(a_np, delta, yaw0)
        fell = is_fallen()
        if fell:
            r -= FALL_PENALTY

        total_r += r

        # Store transition only during training
        if not deterministic:
            buf.store(s, a_np, log_prob.item(), r, value.item(), float(fell))

        s = get_state(delta)   # delta is part of the next state

        # ── Render ────────────────────────────────────────────────────────
        if render:
            _, _, yaw = orientation()
            cam.lookat[0] = mjdata.qpos[0]
            cam.lookat[1] = mjdata.qpos[1]
            # Camera follows from behind (third-person view).
            cam.azimuth = (math.degrees(yaw) + 180) % 360
            renderer.update_scene(mjdata, cam)
            frames.append(renderer.render().copy())

        if fell:
            break

    if renderer:
        renderer.close()

    dist = mjdata.qpos[0] - x0

    # Bootstrap value for GAE: if episode ended naturally (not a fall),
    # the critic's estimate of the last state is a better baseline than 0.
    if deterministic or is_fallen():
        last_val = 0.0
    else:
        with torch.no_grad():
            last_val = ac.vf(
                torch.FloatTensor(s).unsqueeze(0).to(device)
            ).item()

    return total_r, dist, frames, last_val

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 8 — Training loop
# ─────────────────────────────────────────────────────────────────────────────

N_EP         = 5000     # total training episodes
RENDER_EVERY = 200      # record a video every N episodes

# Tracking metrics
reward_history = []
dist_history   = []
best_dist      = -np.inf
best_weights   = None
video_count    = 0

print(f"\nTraining for {N_EP} episodes...\n")

for ep in trange(N_EP):

    # ── Collect one training episode ──────────────────────────────────────
    r, d, _, last_val = run_episode(deterministic=False)
    reward_history.append(r)
    dist_history.append(d)

    # ── Track best policy ─────────────────────────────────────────────────
    if d > best_dist:
        best_dist    = d
        best_weights = {k: v.clone() for k, v in ac.state_dict().items()}

    # ── PPO update (once enough steps accumulated) ────────────────────────
    if len(buf) >= N_STEPS_PER_UPDATE:
        ppo_update(last_val)

    # ── Console log every 10 episodes ────────────────────────────────────
    if ep % 10 == 0:
        avg10 = np.mean(dist_history[-10:])
        std   = math.exp(ac.log_std.data.mean().item())
        print(f"  ep {ep:5d} | dist={d:.3f}m (avg10={avg10:.3f}) | "
              f"reward={r:.1f} | buf={len(buf)} | σ={std:.3f} | best={best_dist:.3f}m")

    # ── Record video with best policy ────────────────────────────────────
    if ep > 0 and ep % RENDER_EVERY == 0 and best_weights:
        # Temporarily swap in the best weights.
        current_w = {k: v.clone() for k, v in ac.state_dict().items()}
        ac.load_state_dict(best_weights)

        _, bd, frames, _ = run_episode(deterministic=True, render=True)
        if frames:
            path = f"movies/rollout_{video_count:03d}_ep{ep:04d}_{bd:.2f}m.mp4"
            media.write_video(path, frames, fps=30)
            print(f"  → Saved {path}")
        video_count += 1

        ac.load_state_dict(current_w)

    # ── Checkpoint every 200 episodes ────────────────────────────────────
    if ep % 200 == 199:
        torch.save({
            "ac":        ac.state_dict(),
            "best_w":    best_weights,
            "best_dist": best_dist,
            "ep":        ep,
        }, f"models/checkpoint_ep{ep}.pth")

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 9 — Final evaluation + plots
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n=== Training done.  Best distance: {best_dist:.3f} m ===\n")

if best_weights:
    # Save the best policy.
    torch.save({"ac": best_weights, "best_dist": best_dist}, "models/best.pth")
    print("Saved: models/best.pth")

    # Record a final video.
    ac.load_state_dict(best_weights)
    _, bd, frames, _ = run_episode(deterministic=True, render=True)
    if frames:
        media.write_video("movies/final.mp4", frames, fps=30)
        print(f"Saved: movies/final.mp4  (distance = {bd:.3f} m)")

# Learning curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
w = 20   # smoothing window
smooth = lambda x: np.convolve(x, np.ones(w) / w, "valid") if len(x) >= w else x

axes[0].plot(smooth(dist_history),   color="#2ecc71", lw=1.5)
axes[0].set_title("Distance walked per episode (m)")
axes[0].set_xlabel("Episode")
axes[0].grid(True, alpha=0.3)

axes[1].plot(smooth(reward_history), color="#3498db", lw=1.5)
axes[1].set_title("Total reward per episode")
axes[1].set_xlabel("Episode")
axes[1].grid(True, alpha=0.3)

plt.suptitle("PPO Training — Quadruped Locomotion", fontsize=13)
plt.tight_layout()
plt.savefig("models/learning_curves.png", dpi=120)
plt.show()
print("Saved: models/learning_curves.png")
