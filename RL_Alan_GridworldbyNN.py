# gridworld_report.py
from __future__ import annotations
import random
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

# ---------- Env ----------
@dataclass(frozen=True)
class Pos:
    r: int
    c: int

class Gridworld3x4:
    n_rows, n_cols = 3, 4
    A_N, A_E, A_S, A_W = 0, 1, 2, 3
    actions = [A_N, A_E, A_S, A_W]
    aname = {A_N: "N", A_E: "E", A_S: "S", A_W: "W"}
    delta = {A_N: (1, 0), A_E: (0, 1), A_S: (-1, 0), A_W: (0, -1)}
    left  = {A_N: A_W, A_E: A_N, A_S: A_E, A_W: A_S}
    right = {A_N: A_E, A_E: A_S, A_S: A_W, A_W: A_N}

    def __init__(self):
        self.start = Pos(0, 0)
        self.blocked = {Pos(1, 1)}
        self.terms = {Pos(2, 3): 1.0, Pos(1, 3): -1.0}
        self.step_r = -0.04
        self.s = self.start

    def reset(self):
        self.s = self.start
        return self.s

    def _in(self, p: Pos) -> bool:
        return 0 <= p.r < self.n_rows and 0 <= p.c < self.n_cols

    def _move(self, s: Pos, a: int) -> Pos:
        dr, dc = self.delta[a]
        n = Pos(s.r + dr, s.c + dc)
        return s if (not self._in(n) or n in self.blocked) else n

    def step(self, a: int):
        s = self.s
        if s in self.terms:
            return s, self.terms[s], True, {}
        u = random.random()
        seq = [(0.8, a), (0.1, self.left[a]), (0.1, self.right[a])]
        acc = 0.0
        chosen = a
        for p, aa in seq:
            acc += p
            if u <= acc:
                chosen = aa
                break
        ns = self._move(s, chosen)
        if ns in self.terms:
            r, done = self.terms[ns], True
        else:
            r, done = self.step_r, False
        self.s = ns
        return ns, r, done, {}

# ---------- Tiny NN for Q(s,a) ----------
class TinyQ:
    def __init__(self, hidden=64, lr=0.02, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.uniform(-1, 1, (2, hidden)) * np.sqrt(6 / (2 + hidden))
        self.b1 = np.zeros((hidden,))
        self.W2 = rng.uniform(-1, 1, (hidden, 4)) * np.sqrt(6 / (hidden + 4))
        self.b2 = np.zeros((4,))
        self.lr = lr

    @staticmethod
    def relu(x): return np.maximum(x, 0.0)

    @staticmethod
    def relu_g(x): return (x > 0).astype(float)

    @staticmethod
    def enc(pos: Pos):
        return np.array([pos.r / 2.0, pos.c / 3.0], dtype=float)

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        h = self.relu(z1)
        z2 = h @ self.W2 + self.b2
        return z2, (x, z1, h, z2)

    def q(self, s: Pos):
        x = self.enc(s)
        q, _ = self.forward(x)
        return q

    def update(self, s: Pos, a: int, target: float):
        x = self.enc(s)
        q, (x, z1, h, z2) = self.forward(x)
        g2 = np.zeros_like(q)
        g2[a] = q[a] - target
        dW2 = np.outer(h, g2)
        db2 = g2
        dh = g2 @ self.W2.T
        dz1 = dh * self.relu_g(z1)
        dW1 = np.outer(x, dz1)
        db1 = dz1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        return float(0.5 * (q[a] - target) ** 2)

# ---------- Helpers ----------
def eval_policy(env: Gridworld3x4, qnet: TinyQ, episodes=200, seed=0) -> float:
    rs = []
    random.seed(seed)
    for _ in range(episodes):
        s = env.reset()
        done = False
        R = 0.0
        for _ in range(100):
            if done:
                break
            a = int(np.argmax(qnet.q(s)))
            s, r, done, _ = env.step(a)
            R += r
        rs.append(R)
    return float(np.mean(rs))

def moving_avg(x, w=50):
    x = np.array(x, dtype=float)
    if len(x) < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / w

def train_once(episodes=2800, hidden=64, lr=0.02, eps_hi=1.0, eps_lo=0.05, decay=2000, seed=11, log_eval_every=50):
    random.seed(seed)
    np.random.seed(seed)
    env = Gridworld3x4()
    q = TinyQ(hidden=hidden, lr=lr, seed=seed)
    ep_returns = []
    ep_losses = []
    eval_x = []
    eval_y = []
    steps = 0
    for ep in range(episodes):
        s = env.reset()
        done = False
        R = 0.0
        for _ in range(100):
            if done:
                break
            eps = max(eps_lo, eps_hi - (eps_hi - eps_lo) * steps / decay)
            greedy_a = int(np.argmax(q.q(s)))
            a = greedy_a if random.random() >= eps else random.choice(env.actions)
            ns, r, done, _ = env.step(a)
            target = r if done else (r + 0.99 * np.max(q.q(ns)))
            loss = q.update(s, a, target)
            ep_losses.append(loss)
            R += r
            s = ns
            steps += 1
        ep_returns.append(R)
        if (ep + 1) % log_eval_every == 0:
            score = eval_policy(Gridworld3x4(), q, 200, seed + ep)
            eval_x.append(ep + 1)
            eval_y.append(score)
    return dict(env=env, q=q, returns=ep_returns, losses=ep_losses, eval_x=eval_x, eval_y=eval_y)

def policy_table(env: Gridworld3x4, q: TinyQ):
    table = []
    for r in reversed(range(env.n_rows)):
        row = []
        for c in range(env.n_cols):
            p = Pos(r, c)
            if p in env.blocked:
                row.append("X")
                continue
            if p in env.terms:
                row.append("T")
                continue
            a = int(np.argmax(q.q(p)))
            row.append(env.aname[a])
        table.append(row)
    return table

# ---------- Main ----------
if __name__ == "__main__":
    base = train_once()

    plt.figure()
    plt.plot(base["returns"], label="episode return")
    ma = moving_avg(base["returns"], 50)
    start_idx = len(base["returns"]) - len(ma)
    x = np.arange(start_idx, start_idx + len(ma))
    plt.plot(x, ma, label="moving average (w=50)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Training Return (Gridworld 3x4)")
    plt.legend()
    plt.savefig(HERE / "gridworld_return.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(base["eval_x"], base["eval_y"], marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Average return (200 eval eps)")
    plt.title("Evaluation Curve (Greedy Policy)")
    plt.savefig(HERE / "gridworld_eval.png", dpi=200)
    plt.close()

    plt.figure()
    loss_ma = moving_avg(base["losses"], 200)
    plt.plot(loss_ma)
    plt.xlabel("Update step")
    plt.ylabel("TD loss (moving avg, w=200)")
    plt.title("Training TD Loss")
    plt.savefig(HERE / "gridworld_tdloss.png", dpi=200)
    plt.close()

    plt.figure()
    for lr in [0.01, 0.02, 0.05]:
        out = train_once(episodes=1600, lr=lr, seed=21, decay=1500)
        plt.plot(out["eval_x"], out["eval_y"], label=f"lr={lr}")
    plt.xlabel("Episode")
    plt.ylabel("Average return (200 eval eps)")
    plt.title("Effect of Learning Rate")
    plt.legend()
    plt.savefig(HERE / "gridworld_lr_sweep.png", dpi=200)
    plt.close()

    pol = policy_table(base["env"], base["q"])
    with open(HERE / "gridworld_policy.txt", "w") as f:
        for row in pol:
            f.write(" ".join(row) + "\n")
