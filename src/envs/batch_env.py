from stable_baselines3.common.vec_env import VecEnvWrapper

class BatchVecEnvWrapper(VecEnvWrapper):
    def __init__(self, batch_env):
        self.env = batch_env
        super().__init__(batch_env)
    
    def reset(self):
        return self.env.reset()
    
    def step_async(self, actions):
        self._actions = actions
    
    def step_wait(self):
        return self.env.step(self._actions)
