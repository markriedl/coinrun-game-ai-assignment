import numpy as np
from coinrun import setup_utils, make
import coinrun.main_utils as utils
from coinrun.config import Config
from gym.envs.classic_control import rendering
from coinrun import policies, wrappers

def random_agent(num_envs=1, max_steps=100000):
    setup_utils.setup_and_load(use_cmd_line_args=True)
    print(Config.IS_HIGH_RES)
    env = make('standard', num_envs=num_envs)
    env.render()
    viewer = rendering.SimpleImageViewer()
    for step in range(max_steps):
        acts = np.array([env.action_space.sample() for _ in range(env.num_envs)])
        _obs, rews, _dones, _infos = env.step(acts)
        print("step", step, "rews", rews)
        env.render()
    env.close()


if __name__ == '__main__':
    random_agent()