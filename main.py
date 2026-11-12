"""Multi-ghost busting with a Naive Bayes sensor model and a coarse-to-fine footprint.

Model (Ghostbusters-style, per cell):
  G_c in {0,1}   : is there a ghost in cell c?  (prior = NUM_OBJECTS / cells)
  D              : binary detector reading over the square sensor footprint F
  P(D=1 | ghost in F)    = hit(s)   (drops toward 0.5-ish as footprint size s grows)
  P(D=1 | no ghost in F) = fa(s)    (false-alarm rate grows with s)
Larger footprint => noisier reading, exactly like the far-away colors in Ghostbusters.

Naive Bayes update: the reading is treated as evidence for each cell in the footprint,
P(G_c | D) ∝ P(G_c) P(D | G_c), where P(D | G_c) marginalises the *other* cells in the
footprint using their current beliefs (cells are independent given the evidence model).

Policy: footprint size follows a schedule (large -> small). At each size the robot goes to
the window with the best expected information gain per unit travel; when no window of that
size is informative any more, it shrinks the footprint. Stops once NUM_OBJECTS cells are
confidently localised.
"""
import os
import random

import matplotlib
if os.environ.get("NO_ANIM"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np

GRID_SIZE = 50
NUM_OBJECTS = 10
SIZE_SCHEDULE = [16, 8, 4, 2, 1]   # sensor footprint side length, large -> small
IG_THRESHOLD = 0.03                # bits; shrink footprint when best window is less informative
CONFIDENT = 0.97                   # belief needed to call a cell "a ghost"
MAX_STEPS = 3000
ANIMATE = not os.environ.get("NO_ANIM")
SEED = None                        # set an int for reproducible runs

EPS = 1e-6


# ---------------------------------------------------------------- sensor model
def hit_prob(s):
    """P(D=1 | ghost in footprint): less reliable as footprint grows."""
    return max(0.60, 0.95 - 0.02 * (s - 1))


def false_alarm(s):
    """P(D=1 | no ghost in footprint): more false alarms as footprint grows."""
    return min(0.30, 0.02 + 0.015 * (s - 1))


def entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def window_sums(a, s):
    """Sum of `a` over every s x s window. Result indexed [x0, y0]. a is indexed [x, y]."""
    c = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
    c[1:, 1:] = a.cumsum(0).cumsum(1)
    return c[s:, s:] - c[:-s, s:] - c[s:, :-s] + c[:-s, :-s]


def sense(truth, x0, y0, s):
    """Simulated noisy reading of the footprint with lower-left cell (x0, y0)."""
    present = truth[x0:x0 + s, y0:y0 + s].any()
    return random.random() < (hit_prob(s) if present else false_alarm(s))


# ---------------------------------------------------------------- naive Bayes inference
def bayes_update(belief, x0, y0, s, d):
    """Update belief (indexed [x, y]) in place after reading d over the footprint."""
    h, f = hit_prob(s), false_alarm(s)
    b = np.clip(belief[x0:x0 + s, y0:y0 + s], EPS, 1 - EPS)
    q_all = np.exp(np.log(1 - b).sum())      # P(no ghost anywhere in footprint)
    r = np.clip(q_all / (1 - b), 0, 1)       # P(no ghost in the *other* cells)
    if d:
        like1 = h
        like0 = f * r + h * (1 - r)
    else:
        like1 = 1 - h
        like0 = (1 - f) * r + (1 - h) * (1 - r)
    post = b * like1 / (b * like1 + (1 - b) * like0)
    belief[x0:x0 + s, y0:y0 + s] = np.clip(post, EPS, 1 - EPS)


def best_window(belief, s, pos):
    """Window of size s maximising expected information gain / travel cost."""
    q = np.exp(window_sums(np.log(1 - np.clip(belief, EPS, 1 - EPS)), s))
    px = 1 - q
    h, f = hit_prob(s), false_alarm(s)
    pd = px * h + (1 - px) * f
    ig = entropy(pd) - px * entropy(h) - (1 - px) * entropy(f)
    xs, ys = np.meshgrid(np.arange(ig.shape[0]), np.arange(ig.shape[1]), indexing="ij")
    dist = np.maximum(abs(xs - pos[0]), abs(ys - pos[1]))
    score = ig / (1 + 0.05 * dist)
    i = np.unravel_index(np.argmax(score), score.shape)
    return int(i[0]), int(i[1]), float(ig[i])


def finished(belief):
    conf = belief > CONFIDENT
    return conf.sum() >= NUM_OBJECTS and belief[~conf].sum() < 1.0


# ---------------------------------------------------------------- visualisation
class View:
    def __init__(self, truth):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.img = self.ax.imshow(np.full((GRID_SIZE, GRID_SIZE), NUM_OBJECTS / GRID_SIZE ** 2).T,
                                  origin="lower", extent=(0, GRID_SIZE, 0, GRID_SIZE),
                                  cmap="jet", vmin=0, vmax=1)
        self.fig.colorbar(self.img, ax=self.ax, label="P(ghost in cell)", fraction=0.046)
        self.ax.add_patch(Rectangle((0, 0), GRID_SIZE, GRID_SIZE, fill=False, edgecolor="red"))
        # true ghosts (hidden info, drawn small & faint so you can compare with the belief)
        for x, y in zip(*np.nonzero(truth)):
            self.ax.add_patch(Circle((x + .5, y + .5), 0.35, fill=False, edgecolor="white", lw=0.8))
        self.found = []
        self.rect = self.ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor="cyan", lw=2))
        self.ax.set_xlim(-1, GRID_SIZE + 1)
        self.ax.set_ylim(-1, GRID_SIZE + 1)
        self.ax.set_aspect("equal")

    def draw(self, belief, pos, s, title):
        self.img.set_data(belief.T)
        self.rect.set_xy(pos)
        self.rect.set_width(s)
        self.rect.set_height(s)
        self.ax.set_title(title)
        plt.pause(0.001)

    def mark_found(self, belief):
        for p in self.found:
            p.remove()
        self.found = [self.ax.add_patch(Circle((x + .5, y + .5), 0.45, color="red", alpha=0.7))
                      for x, y in zip(*np.nonzero(belief > CONFIDENT))]


# ---------------------------------------------------------------- main loop
def main():
    if SEED is not None:
        random.seed(SEED)
    truth = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    for idx in random.sample(range(GRID_SIZE * GRID_SIZE), NUM_OBJECTS):
        truth[idx // GRID_SIZE, idx % GRID_SIZE] = True

    belief = np.full((GRID_SIZE, GRID_SIZE), NUM_OBJECTS / GRID_SIZE ** 2)
    view = View(truth) if ANIMATE else None
    pos = (0, 0)
    steps = measurements = travel = 0
    stage = 0

    while steps < MAX_STEPS and not finished(belief):
        s = SIZE_SCHEDULE[stage]
        x0, y0, ig = best_window(belief, s, pos)
        if ig < IG_THRESHOLD and stage < len(SIZE_SCHEDULE) - 1:
            stage += 1                      # footprint too uninformative: shrink it
            continue

        # drive to the window one 8-neighbour move at a time (like the original controls)
        while pos != (x0, y0):
            pos = (pos[0] + int(np.sign(x0 - pos[0])), pos[1] + int(np.sign(y0 - pos[1])))
            travel += 1
            if ANIMATE and travel % 6 == 0:
                view.draw(belief, pos, s, f"moving... footprint {s}x{s}")

        d = sense(truth, x0, y0, s)
        bayes_update(belief, x0, y0, s, d)
        measurements += 1
        steps += 1
        if ANIMATE:
            view.mark_found(belief)
            view.draw(belief, pos, s,
                      f"meas {measurements} | footprint {s}x{s} | reading={'HIT' if d else 'none'} "
                      f"| confident ghosts {(belief > CONFIDENT).sum()}/{NUM_OBJECTS}")

    est = belief > CONFIDENT
    correct = (est & truth).sum()
    print(f"measurements={measurements} travel={travel} found={est.sum()} "
          f"correct={correct}/{NUM_OBJECTS} false_positives={(est & ~truth).sum()}")
    if ANIMATE:
        view.mark_found(belief)
        view.draw(belief, pos, SIZE_SCHEDULE[stage],
                  f"DONE: {correct}/{NUM_OBJECTS} ghosts busted in {measurements} measurements")
        plt.show()


if __name__ == "__main__":
    main()
