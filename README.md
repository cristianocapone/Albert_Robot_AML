# Albert Robot — Advanced Machine Learning

A hands-on introduction to **deep reinforcement learning for robot locomotion**, built around a simulated 4-legged robot (quadruped) trained with **Proximal Policy Optimization (PPO)**.

Developed for the Advanced Machine Learning course.

---

## What this teaches

| Concept | Where |
|---|---|
| Markov Decision Process | Section 2 & 3 |
| Reward shaping | Section 4 |
| Actor-Critic architecture | Section 5 |
| Gaussian policy + log-probability | Section 5 |
| Generalized Advantage Estimation (GAE) | Section 6 |
| PPO clipped surrogate objective | Section 7 |
| ΔΔθ acceleration-level joint control | Section 3 |

---

## Files

```
Albert_Robot_AML/
├── robot_ppo_tutorial.ipynb   ← main teaching notebook (recommended)
├── robot_ppo_tutorial.py      ← same content as a plain Python script
├── dog.xml                    ← MuJoCo robot description
└── meshes/                    ← robot 3-D geometry (used by dog.xml)
```

---

## Setup

### 1. Create a conda environment

```bash
conda create -n aml python=3.11 -y
conda activate aml
```

### 2. Install dependencies

```bash
pip install torch torchvision          # neural networks
pip install mujoco                     # physics simulator
pip install mediapy                    # video I/O in notebooks
pip install matplotlib tqdm numpy
```

> **Apple Silicon (M1/M2/M3):** MuJoCo works natively on arm64. No extra steps needed.  
> **Linux with GPU:** add `pip install torch --index-url https://download.pytorch.org/whl/cu121` for CUDA support.

### 3. Register the kernel (Jupyter only)

```bash
pip install ipykernel
python -m ipykernel install --user --name aml --display-name "AML"
```

---

## Running

### Option A — Jupyter notebook (recommended for class)

```bash
jupyter lab robot_ppo_tutorial.ipynb
```

Run cells top-to-bottom. Each section has a markdown explanation followed by the code.

### Option B — Plain Python script

```bash
python robot_ppo_tutorial.py
```

---

## Expected output

Training runs for **5000 episodes** (~27 min on CPU, ~5 min on GPU).

```
Control rate : 100 Hz  (dt = 10.0 ms)
Steps/episode: 800
Using: cpu

  ep     0 | dist=0.021m (avg10=0.021) | reward=-12.3 | buf=800  | σ=1.649 | best=0.021m
  ep    10 | dist=0.134m (avg10=0.089) | reward=18.4  | buf=800  | σ=1.612 | best=0.188m
  ...
  ep  4990 | dist=0.743m (avg10=0.710) | reward=80.7  | buf=0    | σ=0.697 | best=1.091m

=== Best distance: 1.091 m ===
```

Outputs are written to:
- `models/best.pth` — saved checkpoint
- `movies/final.mp4` — rendered video of the best policy
- `models/learning_curves.png` — distance and reward over training

---

## Architecture

```
State (24)  =  joint positions (8)  +  joint velocities (8)  +  delta buffer (8)
                         │
              ┌──────────┴──────────┐
              │      ActorCritic     │
              │                      │
              │  Actor  (24→64→8)   │  → Gaussian action → ΔΔθ joint command
              │  Critic (24→256→1)  │  → value estimate V(s)
              └──────────────────────┘
```

### ΔΔθ control (acceleration-level)

The network outputs a small *change* to a velocity buffer, not a direct position command:

```
delta_new = clip( delta + action × MAX_DDELTA,  −DELTA_LIMIT, +DELTA_LIMIT )
ctrl_new  = clip( ctrl  + delta_new,             joint_lo,     joint_hi     )
```

This gives the gait natural inertia and smoothness without any explicit frequency constraint.

---

## Tuning tips

| Goal | What to change |
|---|---|
| Faster walking | increase `TARGET_VX` |
| Smoother gait | increase `ACTION_REG_W` or `DELTA_REG_W` |
| More exploration | increase `LOG_STD_INIT` or `ENTROPY_COEF` |
| Faster convergence | increase `N_STEPS_PER_UPDATE` |
| Straighter path | increase `YAW_W` |

---

## Citation / Credits

Robot model and training environment: Albert project, 2026.  
Algorithm: [Proximal Policy Optimization (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)  
Physics: [MuJoCo](https://mujoco.org/)
