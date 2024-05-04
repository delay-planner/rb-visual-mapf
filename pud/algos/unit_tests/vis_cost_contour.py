import numpy as np
import argparse

import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
from pud.envs.safe_pointenv.safe_pointenv import SafePointEnv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maze_name",
        type=str,
        default="CentralObstacle",
        help="maze name")
    parser.add_argument("--cost_name",
        type=str,
        default="linear",
        help="cost function name")
    parser.add_argument("--radius",
        type=float,
        default=25,
        help="cost function radius from obstacle")
    parser.add_argument("--cost_limit",
        type=float,
        default=2.0,
        help="cost limit")
    parser.add_argument("--resize_factor",
        type=int,
        default=5,
        help="maze resize factor")
    
    args = parser.parse_args()

    env = SafePointEnv(
        walls=args.maze_name,
        resize_factor=args.resize_factor,
        thin=False,
        cost_limit=args.cost_limit,
        cost_f_args={
            "name": args.cost_name,
            "radius": args.radius,
        })

    cost_map  = env.get_cost_map()
    width, height = cost_map.shape

    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots()
    CS = ax.contourf(X, Y, cost_map, cmap=mpl.colormaps["cool"])
    ax.set_title("cost contour with radius = {:.2f}".format(args.radius))
    fig.colorbar(CS)
    fig.savefig("temp/cost_f_contour_{}_{}_r={}.jpg".format(args.maze_name, args.cost_name, args.radius), dpi=300)
    plt.close(fig)
