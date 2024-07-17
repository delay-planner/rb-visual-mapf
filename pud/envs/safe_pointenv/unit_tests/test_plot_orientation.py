import matplotlib.pyplot as plt
from pud.envs.safe_pointenv.safe_pointenv import plot_safe_walls
import numpy as np
from pathlib import Path

L = np.array([[0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0],
              [0, 0, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0]])


out_dir = Path("pud/envs/safe_pointenv/unit_tests/outputs")
out_dir.mkdir(parents=True, exist_ok=True)
L_reading = np.loadtxt("pud/envs/safe_pointenv/unit_tests/L_reading.txt", delimiter=",")

L90 = np.rot90(L)

L270 = np.rot90(L, k=3)

walls = L270
fname = "L270.jpg"

walls = L
fname = "L.jpg"

fname = out_dir.joinpath(fname)

fig, ax = plt.subplots()

ax = plot_safe_walls(walls=walls, cost_map=None, cost_limit=None, ax=ax)
ax.plot(L_reading[:,0], L_reading[:,1], "b-o")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal', adjustable='box')
fig.savefig(fname=fname, dpi=320, bbox_inches="tight")
plt.close(fig)
