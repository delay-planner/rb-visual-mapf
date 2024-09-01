import tempfile
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from typing import Optional
import torch
from tensordict import TensorDict
import numpy as np

class LargeReplayBuffer:
    def __init__(self, 
                max_size=int(1e6),
                batch_size:Optional[int]=None,
                **kwargs,
                ):

        tempdir = tempfile.TemporaryDirectory()
        storage = LazyMemmapStorage(max_size, scratch_dir=tempdir)
        rb_kwargs = dict(storage=storage)
        self.batch_size = None
        if batch_size is not None:
            rb_kwargs["batch_size"] = batch_size
            self.batch_size = batch_size
        self.buffer = TensorDictReplayBuffer(**rb_kwargs)
    
    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, next_state, reward, done):
        new_data = TensorDict({
            "observation": torch.from_numpy(state['observation']),
            "goal": torch.from_numpy(state['goal']),
            "next_observation": torch.from_numpy(next_state['observation']),
            "next_goal": torch.from_numpy(next_state['goal']),
            "action": torch.from_numpy(action),
            "reward": torch.Tensor([reward]),
            "done": torch.Tensor([done]),
        })
        self.buffer.add(new_data)

    def sample(self, batch_size:Optional[int]=None):
        if batch_size is None:
            assert self.batch_size is None
        out = self.buffer.sample(batch_size=batch_size)
        batch = (
            dict(
                observation=out["observation"],
                goal=out["goal"], 
            ),
            dict(
                observation=out["next_observation"],
                goal=out["next_goal"], 
            ),
            out["action"],
            out["reward"],
            out["done"],
        )
        return batch

class ConstrainedLargeReplayBuffer (LargeReplayBuffer):
    def __init__(self, max_size=int(1e6), batch_size:Optional[int]=None, **kwargs):
        super(ConstrainedLargeReplayBuffer, self).__init__(
            max_size=max_size, batch_size=batch_size, **kwargs
        )
    
    def add(self, state, action, next_state, reward, cost, done):
        new_data = TensorDict({
            "observation": torch.from_numpy(state['observation']),
            "goal": torch.from_numpy(state['goal']),
            "next_observation": torch.from_numpy(next_state['observation']),
            "next_goal": torch.from_numpy(next_state['goal']),
            "action": torch.from_numpy(action),
            "reward": torch.Tensor([reward]),
            "done": torch.Tensor([done]),
            "cost": torch.Tensor([cost]),
        })
        self.buffer.add(new_data)

    def sample_w_cost(self, batch_size:Optional[int]=None):
        if batch_size is None:
            assert self.batch_size is None
        out = self.buffer.sample(batch_size=batch_size)
        batch = (
            dict(
                observation=out["observation"],
                goal=out["goal"], 
            ),
            dict(
                observation=out["next_observation"],
                goal=out["next_goal"], 
            ),
            out["action"],
            out["reward"],
            out["cost"],
            out["done"],
        )
        return batch

if __name__ == "__main__":
    rb = LargeReplayBuffer(obs_dim=12, goal_dim=12, action_dim=4, max_size=4)
    for _ in range(10):
        rb.add(
            state={"observation":np.random.rand(32,32).astype(np.int8), 
                "goal":np.random.rand(12).astype(np.int8)},
            action=np.random.rand(4),
            next_state={"observation":np.random.rand(12).astype(np.int8), 
                        "goal":np.random.rand(12).astype(np.int8),},
            reward=-1,
            done=False,
        )
    rb.sample(5)