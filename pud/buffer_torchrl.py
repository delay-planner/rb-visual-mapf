import tempfile
from torchrl.data import ReplayBuffer, LazyMemmapStorage, TensorDictReplayBuffer
from typing import Optional
import torch
from tensordict import TensorDict
import numpy as np

import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()  # blocks execution until client is attached

class ReplayBuffer:
    def __init__(self, 
                obs_dim, 
                goal_dim, 
                action_dim,
                max_size=int(1e6),
                batch_size:Optional[int]=None,
                ):
        super(ReplayBuffer, self).__init__()

        tempdir = tempfile.TemporaryDirectory()
        storage = LazyMemmapStorage(max_size, scratch_dir=tempdir)
        rb_kwargs = dict(storage=storage)
        self.batch_size = None
        if batch_size is not None:
            rb_kwargs["batch_size"] = batch_size
            self.batch_size = batch_size
        self.buffer = TensorDictReplayBuffer(**rb_kwargs)

    def add(self, state, action, next_state, reward, done):
        new_data = TensorDict({
            "s0": torch.FloatTensor(state),
            "a0": torch.FloatTensor(action),
            "s1": torch.FloatTensor(next_state),
            "r": torch.FloatTensor(reward),
            "done": torch.FloatTensor(done),
        })
        self.buffer.add(new_data)

    def sample(self, batch_size:Optional[int]=None):
        if batch_size is None:
            assert self.batch_size is None
        out = self.buffer.sample(batch_size=batch_size)
        import IPython
        IPython.embed(colors="Linux")



if __name__ == "__main__":
    rb = ReplayBuffer(obs_dim=12, goal_dim=12, action_dim=4)
    for _ in range(10):
        rb.add(
            state=np.random.rand(12),
            action=np.random.rand(4),
            next_state=np.random.rand(12),
            reward=np.random.rand(1,),
            done=np.random.rand(1,),
        )
    rb.sample(5)
    #buffer_lazymemmap.extend(data)
    #print(f"The buffer has {len(buffer_lazymemmap)} elements")
    #sample = buffer_lazymemmap.sample()
    #print("sample:", sample)



