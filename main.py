############################################################

def in_ipynb():
  try:
    result = get_ipython().__class__.__name__
    if 'Shell' in result:
      return True
    else:
      return False
  except:
    return False

IN_PYNB = in_ipynb()

#############################################################

import gym
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image 
# pip install Pillow
# pip install torch
# pip install torchvision

from coinrun import setup_utils, make
import coinrun.main_utils as utils
from coinrun.config import Config
if not IN_PYNB:
    from gym.envs.classic_control import rendering
from coinrun import policies, wrappers

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T



import os
import argparse
import pdb

###########################################################

# Seed information
NUM_LEVELS = 1 # repeat the same level over and over
EASY_LEVEL = 1 # Start on a very small map, no enemies
EASY_LEVEL2 = 5
MEDIUM_LEVEL = 15
ONE_MONSTER = 10 # Short map with one monster
HARD_LEVEL = 7 # Longer and with monsters
LAVA_LEVEL = 3 # Longer and with lava and pits

###########################################################
'''
Colab instructions:
New notebook
Edit > Notebook settings > GPU

[1]
import os
del os.environ['LD_PRELOAD']
!apt-get remove libtcmalloc*

[2]
!apt-get update
!apt-get install mpich build-essential qt5-default pkg-config

[3]
import torch
torch.cuda.is_available()

[4]
!git clone https://github.com/markriedl/coinrun-game-ai-assignment.git

[5]
!pip install -r coinrun-game-ai-assignment/requirements.txt

[6]
import sys
sys.path.insert(0, 'coinrun-game-ai-assignment')

[7]
### Testing coinrun with random agent
from coinrun.random_agent import random_agent
random_agent(max_steps=10)

[8]
from main import *
'''



###########################################################
### ARGPARSE

parser = argparse.ArgumentParser(description='Train CoinRun DQN agent.')
parser.add_argument('--render', action="store_true", default=False)
parser.add_argument('--unit_test', action="store_true", default=False)
parser.add_argument('--eval', action="store_true", default=False)
parser.add_argument("--save", help="save the model", default="saved.model")
parser.add_argument("--load", help="load a model", default=None)
parser.add_argument("--episodes", help="number of episodes", type=int, default=1000)
parser.add_argument("--model_path", help="path to saved models", default="saved_models")
parser.add_argument("--seed", help="which level", default=EASY_LEVEL)

args = None
if not IN_PYNB:
    args = parser.parse_args()



###########################################################
### CONSTANTS

# if gpu is to be used
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Resize the screen to this
RESIZE_CONST = 40 

# Defaults
RENDER_SCREEN = args.render if not IN_PYNB else False
SAVE_FILENAME = args.save if not IN_PYNB else 'saved.model'
LOAD_FILENAME = args.load if not IN_PYNB else 'saved.model'
MODEL_PATH = args.model_path if not IN_PYNB else 'saved_models' 
SEED = args.seed if not IN_PYNB else EASY_LEVEL


# Don't play with this
EVAL_EPSILON = 0.1
EVAL_WINDOW_SIZE = 5
EVAL_COUNT = 10
TIMEOUT = 1000
COIN_REWARD = 100

# You may want to change these, but is probably not necessary
BATCH_SIZE = 128
GAMMA = 0.999
BOOTSTRAP = 10000
REPLAY_CAPACITY = 10000
EPSILON = 0.9
EVAL_INTERVAL = 10
NUM_EPISODES = args.episodes if not IN_PYNB else 1000





############################################################
### HELPERS

### Data structure for holding experiences for replay
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

### Function for resizing the screen
resize = T.Compose([T.ToPILImage(),
                    T.Resize(RESIZE_CONST, interpolation=Image.CUBIC),
                    T.ToTensor()])

### Take the environment and return a tensor containing screen data as a 3D tensor containing (color, height, width) information.
### Optional: the screen may be manipulated, for example, it could be cropped
def get_screen(env):
    # Returned screen requested by gym is 512x512x3. Transpose it into torch order (Color, Height, Width).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    ### DO ANY SCREEN MANIPULATIONS NECESSARY (IF ANY)

    ### END SCREEN MANIPULATIONS
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(DEVICE)

### Save the model. Extra information can be added to the end of the filename
def save_model(model, filename, extras = None):
    if extras is not None:
        filename = filename + '.' + str(extras)
    print("Saving", filename, "...")
    torch.save(policy_net, os.path.join(MODEL_PATH, filename))
    print("Done saving.")
 
### Load the model. If there are multiple versions with extra information at the
### end of the filename, get the latest.
def load_model(filename, extras = None):
    if extras is not None:
        filename = filename + '.' + str(extras)
    model = None
    candidates = [os.path.join(MODEL_PATH, f) for f in os.listdir(MODEL_PATH) if filename in f]
    if len(candidates) > 0:
        candidates = sorted(candidates, key=lambda f:os.stat(f).st_mtime, reverse=True)
        filename = candidates[0]
        print("Loading", filename, "...")
        model = torch.load(filename)
        print("Done loading.")
    return model

### Give a text description of the outcome of an episode and also a score
### Score is duration, unless the agent died.
def episode_status(duration, reward):
    status = ""
    score = 0
    if duration >= TIMEOUT:
        status = "timeout"
        score = duration
    elif reward < COIN_REWARD:
        status = "died"
        score = TIMEOUT
    else:
        status = "coin"
        score = duration
    return status, score

############################################################
### ReplayMemory

### Store transitions to use to prevent catastrophic forgetting.
### ReplayMemory implements a ring buffer. Items are placed into memory
###    until memory reaches capacity, and then new items start replacing old items
###    at the beginning of the array. 
### Member variables:
###    capacity: (int) number of transitions that can be stored
###    memory: (array) holds transitions (state, action, next_state, reward)
###    position: (int) index of current location in memory to place the next transition.
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    ### Store a transition in memory.
    ### To implement: put new items at the end of the memory array, unless capacity is reached.
    ###    Combine the arguments into a new Transition object.
    ###    If capacity is reached, start overwriting the beginning of the array.
    ###    Use the position index to keep track of where to put the next item. 
    def push(self, state, action, next_state, reward):
        ### WRITE YOUR CODE BELOW HERE

        ### WRITE YOUR CODE ABOVE HERE
        return None

    ### Return a batch of transition objects from memory containing batch_size elements.
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    ### This allows one to call len() on a ReplayMemory object. E.g. len(replay_memory)
    def __len__(self):
        return len(self.memory)

##########################################################
### DQN

class DQN(nn.Module):

    ### Create all the nodes in the computation graph.
    ### We won't say how to put the nodes together into a computation graph. That is done
    ### automatically when forward() is called.
    def __init__(self, h, w, num_actions):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        ### WRITE YOUR CODE BELOW HERE

        ### WRITE YOUR CODE ABOVE HERE

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        q_values = None
        ### WRITE YOUR CODE BELOW HERE

        ### WRITE YOUR CODE ABOVE HERE
        return q_values

##########################################################
### UNIT TESTING

def testReplayMemory():
    print("Testing ReplayMemory...")
    capacity = 100
    test_replay_memory = ReplayMemory(capacity)
    for i in range(capacity):
        test_replay_memory.push(i, i, i, i)
    assert (len(test_replay_memory) == capacity),"size test failed"
    for i in range(len(test_replay_memory)):
        item = test_replay_memory.memory[i]
        assert (item[0] == i), "item" + str(i) + "not holding the correct value"
    for i in range(capacity//2):
        test_replay_memory.push(capacity+i, capacity+i, capacity+i, capacity+i)
    assert (len(test_replay_memory) == capacity), "size test 2 failed"
    # check items
    for i in range(len(test_replay_memory)):
        item = test_replay_memory.memory[i]
        if i < capacity // 2:
            assert (item[0] == i+capacity), "not holding the correct value after looping (first half)"
        else:
            assert (item[0] == i), "not holding the correct value after looping (second half)"
    print("ReplayMemory test passed.")
    return True


def testMakeBatch():
    print("Testing doMakeBatch...")
    batch_size = 128
    capacity = batch_size * 2
    test_replay_memory = ReplayMemory(capacity)
    state = None
    new_state = None
    action = None
    reward = None
    # Test types and shapes of return values
    for i in range(capacity):
        state = torch.randn(1, 3, 80, 80, device=DEVICE)
        new_state = torch.randn(1, 3, 80, 80, device=DEVICE)
        action = torch.randn(1, 1, device=DEVICE)
        reward = torch.randn(1, 1, device=DEVICE)
        test_replay_memory.push(state, action, new_state, reward)
    states_batch, actions_batch, next_states_batch, rewards_batch, non_final_mask = doMakeBatch(test_replay_memory, batch_size)
    assert(type(states_batch) == torch.Tensor and states_batch.size() == (batch_size, 3, 80, 80)), "states batch not correct shape."
    assert(type(actions_batch) == torch.Tensor and actions_batch.size() == (batch_size, 1)), "actions batch not correct shape."
    assert(type(next_states_batch) == torch.Tensor and next_states_batch.size() == (batch_size, 3, 80, 80)), "next states batch not correct shape."
    assert(type(rewards_batch) == torch.Tensor and rewards_batch.size() == (batch_size, 1)), "rewards batch not correct shape."
    assert(type(non_final_mask) == type(torch.tensor(batch_size, dtype=torch.uint8, device=DEVICE)) and non_final_mask.size()[0] == batch_size), "non-final mask not correct shape."

    # Test mask
    test_replay_memory = ReplayMemory(batch_size)
    for i in range(batch_size):
        state = torch.randn(1, 3, 80, 80)
        new_state = None
        if i % 2 == 0:
            new_state = torch.randn(1, 3, 80, 80, device=DEVICE)
        action = torch.randn(1, 1, device=DEVICE)
        reward = torch.randn(1, 1, device=DEVICE)
        test_replay_memory.push(state, action, new_state, reward)
    states_batch, actions_batch, next_states_batch, rewards_batch, non_final_mask = doMakeBatch(test_replay_memory, batch_size)
    assert(non_final_mask.sum() == batch_size//2), "non_final_mask not masking properly."
    print("doMakeBatch test passed.")
    return True

class UnitTestDQN(nn.Module):
    def __init__(self, h, w, num_actions):
        super(UnitTestDQN, self).__init__()
        self.num_actions = num_actions
    def forward(self, x):
        assert(False), "Network should not be queried when epsilon = 1.0." 
        return None

def testSelectAction():
    print("Testing select_action...")
    sample_size = 10000
    screen_height = 40
    screen_width = 40
    epsilon = 1.0
    num_actions = 7
    net = UnitTestDQN(screen_height, screen_width, num_actions).to(DEVICE)
    state = torch.randn(1, 3, 80, 80, device=DEVICE)
    counts = [0] * num_actions
    for i in range(sample_size):
        action, new_epsilon = select_action(state, net, num_actions, epsilon, steps_done = 0, bootstrap_threshold = 2)
        assert(type(action) == torch.Tensor and action.size() == (1,1)), "Action not correct shape."
        assert(new_epsilon == epsilon), "Epsilon should not change during bootstrapping."
        action = action.item()
        counts[action] = counts[action] + 1
    from scipy.stats import chisquare
    statistic, pvalue = chisquare(counts)
    assert(pvalue > 0.1), "Random sample is not from uniform distribution."    
    print("select_action test passed.")
    return True

def testPredictQValues():
    print("Testing doPredictQValues...")
    batch_size = 128
    screen_height = 80
    screen_width = 80
    num_actions = 7
    net = DQN(screen_height, screen_width, num_actions).to(DEVICE)
    states_batch = torch.randn(batch_size, 3, 80, 80, device=DEVICE)
    actions_batch = torch.randint(0, 7, (128, 1), device=DEVICE)
    state_action_values = doPredictQValues(net, states_batch, actions_batch)
    assert(type(state_action_values) == torch.Tensor and state_action_values.size() == (128, 1)), "Return value not correct shape."
    print("doPredictQValues test passed.")
    return True

def testPredictNextStateUtilities():
    print("Testing doPredictNextStateUtilities...")
    screen_height = 80
    screen_width = 80
    num_actions = 7
    batch_size = 128
    passed = False
    net = DQN(screen_height, screen_width, num_actions).to(DEVICE)
    # First option to try is that the batch is full sized.
    try:
        next_states_batch = torch.ones(batch_size, 3, 80, 80, device=DEVICE)
        non_final_mask = torch.ones(batch_size, dtype=torch.uint8, device=DEVICE)
        for i in range(batch_size):
            if i % 2 == 1:
                next_states_batch[i].fill_(0)
                non_final_mask[i] = 0
        next_state_values = doPredictNextStateUtilities(net, next_states_batch, non_final_mask, batch_size)
        assert(type(next_state_values) == torch.Tensor and next_state_values.size() == (batch_size, 1)), "Return value not correct shape (attempt 1)."
        for i in range(batch_size):
            if i % 2 == 1:
                assert(next_state_values[i].sum() == 0), "Element " + str(i) + "is not 0.0 when non_final_mask[i] = 0"
        passed = True
    except RuntimeError as e:
        print(e)
        print("Will try alternative test.")
    if not passed:
        # Next option is that batch is not full sized.
        try:
            next_states_batch = torch.ones(batch_size-1, 3, 80, 80, device=DEVICE)
            non_final_mask = torch.ones(batch_size, dtype=torch.uint8, device=DEVICE)
            non_final_mask[0] = 0
            next_state_values = doPredictNextStateUtilities(net, next_states_batch, non_final_mask, batch_size)
            assert(type(next_state_values) == torch.Tensor and next_state_values.size()[0] == batch_size), "Return value not correctd shape (attempt 2)."
            passed = True
        except RuntimeError as e:
            print(e)
            print("No further alternative tests available.")
    if passed:
        print("doPredictNextStateUtilities test passed.")
        return True
    assert(False), "doPredictNextStateUtilities did NOT pass test."

def testComputeExpectedQValues():
    print("Testing doComputeExpectedQValues...")
    batch_size = 128
    gamma = 0.5
    next_state_values = torch.ones(batch_size)
    rewards_batch = torch.ones(batch_size)
    expected_state_action_values = doComputeExpectedQValues(next_state_values, rewards_batch, gamma)
    assert(type(expected_state_action_values) == torch.Tensor and expected_state_action_values.size()[0] == batch_size), "Return value not expected shape."
    for i in range(batch_size):
        assert(expected_state_action_values[i] == 1.5), "Element " + str(i) + " doesn't have the correct value."
    print("doComputeExpectedQValues test passed.")
    return True

def testComputeLoss():
    print("Testing doComputeLoss...")
    batch_size = 128
    state_action_values = torch.randn(batch_size, device=DEVICE)
    expected_state_action_values = torch.randn(batch_size, device=DEVICE)
    loss = doComputeLoss(state_action_values, expected_state_action_values)
    assert(type(loss) == torch.Tensor and len(loss.size()) == 0), "Loss not of expected shape."
    print("doComputeLoss test passed.")
    return True


def unit_test():
    testReplayMemory()
    testMakeBatch()
    testSelectAction()
    testPredictQValues()
    testPredictNextStateUtilities()
    testComputeExpectedQValues()
    testComputeLoss()

##########################################################
### WORKER FUNCTIONS

### Choose and instantiate an optimizer. A default example is given, which you can change.
### Input:
### - parameters: the DQN parameters
### Output:
### - the optimizer object
def initializeOptimizer(parameters):
    optimizer = None
    ### WRITE YOUR CODE BELOW HERE

    ### WRITE YOUR CODE ABOVE HERE
    return optimizer

### Select an action to perform. 
### If a random number [0..1] is greater than epsilon, then query the policy_network,
### otherwise use a random action.
### Inputs:
### - state: a tensor of shape 3 x screen_height x screen_width
### - policy_net: a DQN object
### - num_actions: number of actions available
### - epsilon: float [0..1] indicating whether to choose random or use the network
### - steps_done: number of previously executed steps
### - bootstrap_threshold: number of steps that must be executed before training begins
### This function should return:
### - A tensor of shape 1 x 1 that contains the number of the action to execute
### - The new epsilon value to use next time
def select_action(state, policy_net, num_actions, epsilon, steps_done = 0, bootstrap_threshold = 0):
    action = None
    new_epsilon = epsilon
    ### WRITE YOUR CODE BELOW HERE

    ### WRITE YOUR CODE ABOVE HERE
    return action, new_epsilon

### Ask for a batch of experience replays.
### Inputs:
### - replay_memory: A ReplayMemory object
### - batch_size: size of the batch to return
### Outputs:
### - states_batch: a tensor of shape batch_size x 3 x screen_height x screen_width
### - actions_batch: a tensor of shape batch_size x 1 containing action numbers
### - next_states_batch: a tensor containing screens. 
### - rewards_batch: a tensor of shape batch_size x 1 containing reward values.
### - non_final_mask: a tensor of bytes of length batch_size containing a 0 if the state is terminal or 1 otherwise
def doMakeBatch(replay_memory, batch_size):
    states_batch = None
    actions_batch = None
    next_states_batch = None
    rewards_batch = None
    non_final_mask = None
    ### WRITE YOUR CODE BELOW HERE

    ### WRITE YOUR CODE ABOVE HERE
    return states_batch, actions_batch, next_states_batch, rewards_batch, non_final_mask


### Ask the policy_net to predict the Q value for a batch of states and a batch of actions.
### Inputs:
### - policy_net: the DQN
### - states_batch: a tensor of shape batch_size x 3 x screen_height x screen_width containing screens
### - actions_batch: a tensor of shape batch_size x 1 containing action numbers
### Output:
### - A tensor of shape batch_size x 1 containing the Q-value predicted by the DQN in the position indicated by the action
def doPredictQValues(policy_net, states_batch, actions_batch):
    state_action_values = None
    ### WRITE YOUR CODE BELOW HERE

    ### WRITE YOUR CODE ABOVE HERE
    return state_action_values

### Ask the policy_net to predict the utility of a next_state.
### Inputs:
### - policy_net: The DQN
### - next_states_batch: a tensor of shape batch_size x 3 x screen_height x screen_width
### - non_final_mask: a tensor of length batch_size containing 0 for terminal states and 1 for non-terminal states
### - batch_size: the batch size
### Note: Only run non-terminal states through the policy_net
### Output:
### - A tensor of shape batch_size x 1 containing Q-values
def doPredictNextStateUtilities(policy_net, next_states_batch, non_final_mask, batch_size):
    next_state_values = torch.zeros(batch_size, device=DEVICE)
    ### WRITE YOUR CODE BELOW HERE

    ### WRITE YOUR CODE ABOVE HERE
    return next_state_values.detach()

### Compute the Q-update equation Q(s_t, a_t) = R(s_t+1) + gamma * argmax_a' Q(s_t+1, a')
### Inputs:
### - next_state_values: a tensor of shape batch_size x 1 containing Q values for state s_t+1
### - rewards_batch: a tensor or shape batch_size x 1 containing reward values for state s_t+1
### Output:
### - A tensor of shape batch_size x 1
def doComputeExpectedQValues(next_state_values, rewards_batch, gamma):
    expected_state_action_values = None
    ### WRITE YOUR CODE BELOW HERE

    ### WRITE YOUR CODE ABOVE HERE
    return expected_state_action_values

### Compute the loss
### Inputs:
### - state_action_values: a tensor of shape batch_size x 1 containing Q values
### - expected_state_action_values: a tensor of shape batch_size x 1 containing updated Q values
### Output:
### - A tensor scalar value
def doComputeLoss(state_action_values, expected_state_action_values):
    loss = None
    ### WRITE YOUR CODE BELOW HERE

    ### WRITE YOUR CODE ABOVE HERE
    return loss

### Run backpropagation. Make sure gradients are clipped between -1 and +1.
### Inputs:
### - loss: a tensor scalar
### - parameters: the parameters of the DQN
### There is no output
def doBackprop(loss, parameters):
    ### WRITE YOUR CODE BELOW HERE

    ### WRITE YOUR CODE ABOVE HERE
    return None



#########################################################
### OPTIMIZE

### Take a DQN and do one forward-backward pass.
### Since this is Q-learning, we will run a forward pass to get Q-values for state-action pairs and then 
### give the true value as the Q-values after the Q-update equation.
def optimize_model(policy_net, replay_memory, optimizer, batch_size, gamma):
    if len(replay_memory) < batch_size:
        return
    ### step 1: sample from the replay memory. Get BATCH_SIZE transitions
    ### Step 2: Get a list of non-final next states.
    ###         a. Create a mask, a tensor of length BATCH_SIZE where each element i is 1 if 
    ###            batch.next_state[i] is not None and 0 otherwise.
    ###         b. Create a tensor of shape [BATCH_SIZE, color(3), height, width] by concatenating
    ###            all non-final (not None) batch.next_states together.
    ### Step 3: set up batches for state, action, and reward
    ###         a. Create a tensor of shape [BATCH_SIZE, color(3), height, width] holding states
    ###         b. Create a tensor of shape [BATCH_SIZE, 1] holding actions
    ###         c. Create a tensor of shape [BATCH_SIZE, 1] holding rewards
    states_batch, actions_batch, next_states_batch, rewards_batch, non_final_mask = doMakeBatch(replay_memory, batch_size)

    ### Step 4: Get the action values predicted.
    ###         a. Call policy_net(state_batch) to get a tensor of shape [BATCH_SIZE, NUM_ACTIONS] containing Q-values
    ###         b. For each batch, get the Q-value for the corresponding action in action_batch (hint: torch.gather)
    state_action_values = doPredictQValues(policy_net, states_batch, actions_batch)

    ### Step 5: Get the utility values of next_states.
    next_state_values = doPredictNextStateUtilities(policy_net, next_states_batch, non_final_mask, batch_size)
    
    ### Step 6: Compute the expected Q values.
    expected_state_action_values = doComputeExpectedQValues(next_state_values, rewards_batch, gamma)

    ### Step 7: Computer Huber loss (smooth L1 loss)
    ###         Compare state action values from step 5 to expected state action values from step 7
    loss = doComputeLoss(state_action_values, expected_state_action_values)
    ### Step 8: Back propagation
    ###         a. Zero out gradients
    ###         b. call loss.backward()
    ###         c. Prevent gradient explosion by clipping gradients between -1 and 1
    ###            (hint: param.grad.data is the gradients. See torch.clamp_() )
    ###         d. Tell the optimizer that another step has occurred: optimizer.step()
    if optimizer is not None:
        optimizer.zero_grad()
        doBackprop(loss, policy_net.parameters())
        optimizer.step()

##########################################################
### MAIN


### Training loop.
### Each episode is a game that runs until the agent gets the coin or the game times out.
### Train for a given number of episodes.
def train(num_episodes = NUM_EPISODES, load_filename = None, save_filename = None, eval_interval = EVAL_INTERVAL, replay_capacity = REPLAY_CAPACITY, bootstrap_threshold = BOOTSTRAP, epsilon = EPSILON, eval_epsilon = EVAL_EPSILON, gamma = GAMMA, batch_size = BATCH_SIZE, num_levels = NUM_LEVELS, seed = SEED):
    # Set up the environment
    setup_utils.setup_and_load(use_cmd_line_args=False, is_high_res=True, num_levels=num_levels, set_seed=seed)
    env = make('standard', num_envs=1)
    if RENDER_SCREEN and not IN_PYNB:
        env.render()

    # Reset the environment
    env.reset()

    # Get screen size so that we can initialize layers correctly based on shape returned from AI gym. 
    init_screen = get_screen(env)
    _, _, screen_height, screen_width = init_screen.shape
    print("screen size: ", screen_height, screen_width)

    # Are we resuming from an existing model?
    policy_net = None
    if load_filename is not None and os.path.isfile(os.path.join(MODEL_PATH, load_filename)):
        print("Loading model...")
        policy_net = load_model(load_filename)
        policy_net = policy_net.to(DEVICE)
        print("Done loading.")
    else:
        print("Making new model.")
        policy_net = DQN(screen_height, screen_width, env.NUM_ACTIONS).to(DEVICE)
    # Make a copy of the policy network for evaluation purposes
    eval_net = DQN(screen_height, screen_width, env.NUM_ACTIONS).to(DEVICE)
    eval_net.load_state_dict(policy_net.state_dict())
    eval_net.eval()
    
    # Instantiate the optimizer
    optimizer = None
    if len(list(policy_net.parameters())) > 0:
        optimizer = initializeOptimizer(policy_net.parameters())
    
    # Instantiate the replay memory
    replay_memory = ReplayMemory(replay_capacity)

    steps_done = 0               # How many steps have been run
    eval_window = []             # Keep the last 5 episode durations
    best_window = float('inf')   # The best average window duration to date

    ### Do training until episodes complete or until ^C is pressed
    try: 
        print("training...")
        i_episode = 0            # The episode number
        
        # Stop when we reach max episodes
        while i_episode < num_episodes:
            print("episode:", i_episode, "epsilon:", epsilon)
            max_reward = 0       # The best reward we've seen this episode
            done = False         # Has the game ended (timed out or got the coin)
            episode_steps = 0    # Number of steps performed in this episode
            # Initialize the environment and state
            env.reset()
            
            # Current screen. There is no last screen because we get velocity on the screen itself.
            state = get_screen(env)

            # Do forever until the loop breaks
            while not done:
                # Select and perform an action
                action, epsilon = select_action(state, policy_net, env.NUM_ACTIONS, epsilon, steps_done, bootstrap_threshold)
                steps_done = steps_done + 1
                episode_steps = episode_steps + 1
                
                # for debugging
                if RENDER_SCREEN and not IN_PYNB:
                    env.render() 

                # Run the action in the environment
                if action is not None: 
                    _, reward, done, _ = env.step(np.array([action.item()]))

                    # Record if this was the best reward we've seen so far
                    max_reward = max(reward, max_reward)
                    
                    

                    # Turn the reward into a tensor  
                    reward = torch.tensor([reward], device=DEVICE)

                    # Observe new state
                    current_screen = get_screen(env)

                    # Did the game end?
                    if not done:
                        next_state = current_screen
                    else:
                        next_state = None

                    # Store the transition in memory
                    replay_memory.push(state, action, next_state, reward)

                    # Move to the next state
                    state = next_state

                    # If we are past bootstrapping we should perform one step of the optimization
                    if steps_done > bootstrap_threshold:
                      optimize_model(policy_net, replay_memory, optimizer, batch_size, gamma)
                else:
                    # Do nothing if select_action() is not implemented and returning None
                    env.step(np.array([0]))
                    
                # If we are done, print some statistics
                if done:
                    print("duration:", episode_steps)
                    print("max reward:", max_reward)
                    status, _ = episode_status(episode_steps, max_reward)
                    print("result:", status)
                    print("total steps:", steps_done)
                    
            # Should we evaluate?
            if steps_done > bootstrap_threshold and i_episode > 0 and i_episode % eval_interval == 0:
                test_average_duration = 0       # Track the average eval duration
                test_average_max_reward = 0     # Track the average max reward
                # copy all the weights into the evaluation network
                eval_net.load_state_dict(policy_net.state_dict())
                # Evaluate 10 times
                for _ in range(EVAL_COUNT):
                    # Call the evaluation function
                    test_duration, test_max_reward = evaluate(eval_net, eval_epsilon, env)
                    status, score = episode_status(test_duration, test_max_reward)
                    test_duration = score # Set test_duration to score to factor in death-penalty
                    test_average_duration = test_average_duration + test_duration
                    test_average_max_reward = test_average_max_reward + test_max_reward
                test_average_duration = test_average_duration / 10
                test_average_max_reward = test_average_max_reward / 10
                print("Average duration:", test_average_duration)
                print("Average max reward:", test_average_max_reward)
                # Append to the evaluation window
                if len(eval_window) < EVAL_WINDOW_SIZE:
                    eval_window.append(test_average_duration)
                else:
                    eval_window = eval_window[1:] + [test_average_duration]
                # Compute window average
                window_average = sum(eval_window) / len(eval_window)
                print("evaluation window:", eval_window, "window average:", window_average)
                # If this is the best window average we've seen, save the model
                if len(eval_window) >= EVAL_WINDOW_SIZE and window_average <= best_window:
                    best_window = window_average
                    if save_filename is not None:
                        save_model(policy_net, save_filename, i_episode)
            # Only increment episode number if we are done with bootstrapping
            if steps_done > bootstrap_threshold:
              i_episode = i_episode + 1
        print('Training complete')
    except KeyboardInterrupt:
        print("Training interrupted")
    if RENDER_SCREEN and not IN_PYNB:
        env.render()
    env.close()
    return policy_net
 
 

### Evaluate the DQN
### If environment is given, use that. Otherwise make a new environment.
def evaluate(policy_net, epsilon = EVAL_EPSILON, env = None):
    setup_utils.setup_and_load(use_cmd_line_args=False, is_high_res=True, num_levels=NUM_LEVELS, set_seed=SEED)
    
    # Make an environment if we don't already have one
    if env is None:
        env = make('standard', num_envs=1)
    if RENDER_SCREEN and not IN_PYNB:
        env.render()

    # Reset the environment
    env.reset()

    # Get screen size so that we can initialize layers correctly based on shape
    # returned from AI gym. 
    init_screen = get_screen(env)
    _, _, screen_height, screen_width = init_screen.shape

    # Get the network ready for evaluation (turns off some things like dropout if used)
    policy_net.eval()

    # Current screen. There is no last screen
    state = get_screen(env)

    steps_done = 0         # Number of steps executed
    max_reward = 0         # Max reward seen
    done = False           # Is the game over?

    print("Evaluating...")
    while not done:
        # Select and perform an action
        action, _ = select_action(state, policy_net, env.NUM_ACTIONS, epsilon, steps_done=0, bootstrap_threshold=0)
        steps_done = steps_done + 1

        if RENDER_SCREEN and not IN_PYNB:
            env.render()

        # Execute the action
        if action is not None:
            _, reward, done, _ = env.step(np.array([action.item()]))

            # Is this the best reward we've seen?
            max_reward = max(reward, max_reward)

            # Observe new state
            state = get_screen(env)
        else:
            # Do nothing if select_action() is not implemented and returning None
            env.step(np.array([0]))

    print("duration:", steps_done)
    print("max reward:", max_reward)
    status, _ = episode_status(steps_done, max_reward)
    print("result:", status)
    if RENDER_SCREEN and not IN_PYNB:
        env.render()
    return steps_done, max_reward



if __name__== "__main__":
    if not IN_PYNB:
        if args.unit_test:
            unit_test()
        elif args.eval:
            if args.load is not None and os.path.isfile(os.path.join(MODEL_PATH, args.load)):
                eval_net = load_model(args.load) 
                print(eval_net)
                for _ in range(EVAL_COUNT):
                    evaluate(eval_net, EVAL_EPSILON)
        else:
            policy_net = train()
            for _ in range(EVAL_COUNT):
                evaluate(policy_net, EVAL_EPSILON)
