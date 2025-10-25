# gridworld_nn_q.py
# Q-learning with a tiny NumPy neural net (no external ML libs).
# Grid per prompt: 3x4, terminals: +1 at (row=3,col=4), -1 at (row=2,col=4),
# blocked at (row=2,col=2). Start at (1,1). Rewards: -0.04 per step.
# Stochastic dynamics: intended 0.8, slip-left 0.1, slip-right 0.1.
# Coordinates use bottom-left origin; rows: 1..3 bottom→top, cols: 1..4 left→right.

from __future__ import annotations
import random, numpy as np
from dataclasses import dataclass

# ---------- Environment ----------
@dataclass(frozen=True)
class Pos:
    r: int  # 0=bottom row
    c: int  # 0=left col

class Gridworld3x4:
    n_rows, n_cols = 3, 4
    A_N, A_E, A_S, A_W = 0, 1, 2, 3
    actions = [A_N, A_E, A_S, A_W]
    aname  = {A_N:"N", A_E:"E", A_S:"S", A_W:"W"}
    delta  = {A_N:( 1,0), A_E:(0, 1), A_S:(-1,0), A_W:(0,-1)}
    left   = {A_N:A_W, A_E:A_N, A_S:A_E, A_W:A_S}
    right  = {A_N:A_E, A_E:A_S, A_S:A_W, A_W:A_N}

    def __init__(self):
        self.start   = Pos(0,0)
        self.blocked = {Pos(1,1)}                # (row=2,col=2)
        self.terms   = {Pos(2,3):+1.0, Pos(1,3):-1.0}  # (3,4)=+1, (2,4)=-1
        self.step_r  = -0.04
        self.s = self.start

    def reset(self): self.s = self.start; return self.s
    def _in(self,p): return 0<=p.r<self.n_rows and 0<=p.c<self.n_cols
    def _move(self,s,a):
        dr,dc = self.delta[a]; n = Pos(s.r+dr, s.c+dc)
        return s if (not self._in(n) or n in self.blocked) else n

    def step(self, a):
        s = self.s
        if s in self.terms: return s, self.terms[s], True, {}
        u = random.random()
        seq = [(0.8,a),(0.1,self.left[a]),(0.1,self.right[a])]
        acc=0.0; chosen=a
        for p,aa in seq:
            acc+=p
            if u<=acc: chosen=aa; break
        ns = self._move(s, chosen)
        r, done = (self.terms[ns], True) if ns in self.terms else (self.step_r, False)
        self.s = ns
        return ns, r, done, {}

# ---------- Tiny neural net for Q(s,a) ----------
class TinyQ:
    # 2→H→4 MLP with ReLU, trained by semi-gradient TD
    def __init__(self, hidden=64, lr=0.02, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.uniform(-1,1,(2,hidden)) * (np.sqrt(6/(2+hidden)))
        self.b1 = np.zeros((hidden,))
        self.W2 = rng.uniform(-1,1,(hidden,4)) * (np.sqrt(6/(hidden+4)))
        self.b2 = np.zeros((4,))
        self.lr = lr

    @staticmethod
    def _relu(x): return np.maximum(x,0.0)
    @staticmethod
    def _relu_g(x): return (x>0).astype(float)

    @staticmethod
    def _enc(pos: Pos):
        # normalize row to [0,1] using /2, col to [0,1] using /3
        return np.array([pos.r/2.0, pos.c/3.0], dtype=float)

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        h  = self._relu(z1)
        z2 = h @ self.W2 + self.b2
        return z2, (x,z1,h,z2)

    def q(self, s: Pos):
        x = self._enc(s)
        q,_ = self.forward(x)
        return q

    def update(self, s: Pos, a: int, target: float):
        x = self._enc(s)
        q, (x,z1,h,z2) = self.forward(x)
        # L = 0.5*(q[a]-target)^2
        g2 = np.zeros_like(q); g2[a] = (q[a]-target)           # dL/dz2
        dW2 = np.outer(h, g2); db2 = g2
        dh  = g2 @ self.W2.T
        dz1 = dh * self._relu_g(z1)
        dW1 = np.outer(x, dz1); db1 = dz1
        # SGD
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1

# ---------- Train ----------
def train(seed=123):
    random.seed(seed); np.random.seed(seed)
    env  = Gridworld3x4()
    qnet = TinyQ(hidden=64, lr=0.02, seed=seed)
    gamma = 0.99
    episodes = 8000
    eps_hi, eps_lo, decay = 1.0, 0.05, 6000
    steps = 0

    for _ in range(episodes):
        s = env.reset(); done = False
        for _ in range(100):               # soft horizon
            if done: break
            eps = max(eps_lo, eps_hi - (eps_hi-eps_lo)*steps/decay)
            if random.random() < eps: a = random.choice(env.actions)
            else: a = int(np.argmax(qnet.q(s)))

            ns, r, done, _ = env.step(a)
            target = r if done else (r + gamma * float(np.max(qnet.q(ns))))
            qnet.update(s, a, target)
            s = ns; steps += 1
    return env, qnet

# ---------- Report ----------
def tables(env, qnet):
    policy, values = [], []
    for r in reversed(range(env.n_rows)):      # top row first
        prow, vrow = [], []
        for c in range(env.n_cols):
            p = Pos(r,c)
            if p in env.blocked:    prow.append("###"); vrow.append(None); continue
            if p in env.terms:      prow.append("T");   vrow.append(env.terms[p]); continue
            q = qnet.q(p); a = int(np.argmax(q)); v = float(np.max(q))
            prow.append(env.aname[a]); vrow.append(round(v,3))
        policy.append(prow); values.append(vrow)
    return policy, values

if __name__ == "__main__":
    env, qnet = train(seed=7)
    pol, val = tables(env, qnet)
    print("Policy (top row first):")
    for row in pol: print(row)
    print("\nValues (max_a Q):")
    for row in val: print(row)
