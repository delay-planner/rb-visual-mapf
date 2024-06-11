# (Multi-Agent) Hierarchal Constrained Reinforcement Learning

## Design custom evaluation problems for illustration
**Step 1**: generate a figure of the 2D maze

**Step 2**: load the image into [WebPlotDigitizer](https://apps.automeris.io/wpd/), manually align the x and y axes by selecting the start and end points. 

**Step 3**: pick the start and goal positions from the figure, click "View Data" button, copy the coords to a new file under [illustration_set](pud/envs/safe_pointenv/illustration_set) illustration following the specification [here](pud/envs/safe_pointenv/illustration_set/README.md).


## Train with visual inputs (on merge  branch)
```bash
bash launch_jobs/local_debug_vec_habitat.sh
```
make sure to adjust the number of vector envs depending on your GPU memory and speed.

With vector envs, the habitat environment class is unchanged, but there is a special collector that does batch inference to speed up action decision.

## Installing habitat-sim
### We require python>=3.9 and cmake>=3.10
**Step 1**:
```bash
conda install habitat-sim -c conda-forge -c aihabitat
```
**Step 2**:
### Download (testing) 3D scenes
```bash
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path /path/to/data/
```
### Download example objects
```bash
python -m habitat_sim.utils.datasets_download --uids habitat_example_objects --data-path /path/to/data/
```



# Sparse Graphical Memory (SGM) and Search on the Replay Buffer (SoRB) in PyTorch

## Example usage
```
pip install -e .

python run_PointEnv.py configs/config_PointEnv.py
```

## Results

### SoRB (re-planning with closest waypoint) trajectory visualization
![Search comparison](./workdirs/uvfddpg_FourRooms/sorb_compare_search_openloop0.png)

```
policy: no search
start: [0.03271197 0.99020872]
goal: [0.81310241 0.028764  ]
steps: 300
----------
policy: search
start: [0.03271197 0.99020872]
goal: [0.81310241 0.028764  ]
steps: 127
```

### SoRB (open loop planning) trajectory visualization
![Search comparison](./workdirs/uvfddpg_FourRooms/sorb_compare_search_openloop1.png)

```
policy: no search
start: [0.03271197 0.99020872]
goal: [0.81310241 0.028764  ]
steps: 300
----------
policy: search
start: [0.03271197 0.99020872]
goal: [0.81310241 0.028764  ]
steps: 111
```

### State graph visualization 

1. SoRB state graph (per critic in ensemble)
![SoRB state graph](./workdirs/uvfddpg_FourRooms/sorb_state_graph_ensemble.png)

2. SGM state graph (ensembled)
<!-- ![SGM state graph](./workdirs/uvfddpg_FourRooms/sgm_state_graph.png) -->
<p align="center"><img src="./workdirs/uvfddpg_FourRooms/sgm_state_graph.png" width="275" alt="SGM state graph"></p>

```
Initial SparseSearchPolicy (|V|=202, |E|=1894) has success rate 0.20, evaluated in 14.26 seconds
Filtered SparseSearchPolicy (|V|=202, |E|=986) has success rate 0.80, evaluated in 8.44 seconds
Took 10000 cleanup steps in 84.45 seconds
Cleaned SparseSearchPolicy (|V|=202, |E|=955) has success rate 1.00, evaluated in 6.69 seconds
```

## Credits
* https://github.com/scottemmons/sgm
* https://github.com/google-research/google-research/tree/master/sorb
* https://github.com/sfujim/TD3

## References
[1]: Michael Laskin, Scott Emmons, Ajay Jain, Thanard Kurutach, Pieter Abbeel, Deepak Pathak, ["Sparse Graphical Memory for Robust Planning"](https://arxiv.org/abs/2003.06417), 2020.

[2]: Benjamin Eysenbach, Ruslan Salakhutdinov, Sergey Levine, ["Search on the Replay Buffer: Bridging Planning and Reinforcement Learning"](https://arxiv.org/abs/1906.05253), 2019.

