
import sys
import os
import traceback

SKIP_UNIT_TESTS = False
SKIP_TRAINING = False

is_easy = False
is_medium = False
is_monster = False
is_all = False

if sys.argv[1] == 'easy':
    is_easy = True
elif sys.argv[1] == 'medium':
    is_medium = True
elif sys.argv[1] == 'monster':
    is_monster = True
else:
    is_all = True

sys.argv = sys.argv[:1]

if is_easy:
    print("importing main_easy")
    from main_easy import *
elif is_medium:
    print("importing main_medium")
    from main_medium import *
elif is_monster:
    print("importing main_monster")
    from main_monster import *
else:
    print("importing main")
    from main import *
    is_all = True

EVAL_COUNT = 1
MY_BATCH_SIZE = 2
MODEL_PATH = '.'

unit_test_weight = 2.0
training_weight = 1.0
test_weight = 1.0

unit_test_grade = 0.0
training_grade = 0.0
test_grade = 0.0

def grade_train(num_episodes = NUM_EPISODES, load_filename = None, save_filename = None, eval_interval = EVAL_INTERVAL, replay_capacity = REPLAY_CAPACITY, bootstrap_threshold = BOOTSTRAP, epsilon = EPSILON, eval_epsilon = EVAL_EPSILON, gamma = GAMMA, batch_size = BATCH_SIZE, target_update = TARGET_UPDATE, random_seed = RANDOM_SEED, num_levels = NUM_LEVELS, seed = SEED):
    # Set the random seed
    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RANDOM_SEED)
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
    # Target network is a snapshot of the policy network that lags behind (for stablity)
    target_net = DQN(screen_height, screen_width, env.NUM_ACTIONS).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # Instantiate the optimizer
    optimizer = None
    if len(list(policy_net.parameters())) > 0:
        optimizer = initializeOptimizer(policy_net.parameters())
    
    # Instantiate the replay memory
    replay_memory = ReplayMemory(replay_capacity)

    steps_done = 0               # How many steps have been run
    best_eval = float('inf')     # The best model evaluation to date

    ### Do training until episodes complete 
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
                  optimize_model(policy_net, target_net if target_update > 0 else policy_net, replay_memory, optimizer, batch_size, gamma)
            else:
                # Do nothing if select_action() is not implemented and returning None
                env.step(np.array([0]))
                
            # If we are done, print some statistics
            if done:
                print("duration:", episode_steps)
                print("max reward:", max_reward)
                status, _ = episode_status(episode_steps, max_reward)
                print("result:", status)
                print("total steps:", steps_done, '\n')

            # Should we update the target network?
            if target_update > 0 and i_episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
                
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
            test_average_duration = test_average_duration / EVAL_COUNT
            test_average_max_reward = test_average_max_reward / EVAL_COUNT
            print("Average duration:", test_average_duration)
            print("Average max reward:", test_average_max_reward)
            # If this is the best window average we've seen, save the model
            if test_average_duration < best_eval:
                best_eval = test_average_duration
                if save_filename is not None:
                    save_model(policy_net, save_filename, i_episode)
            print(' ')
        # Only increment episode number if we are done with bootstrapping
        if steps_done > bootstrap_threshold:
          i_episode = i_episode + 1
    print('Training complete')
    if RENDER_SCREEN and not IN_PYNB:
        env.render()
    env.close()
    return policy_net

def grade_evaluate(policy_net, epsilon = EVAL_EPSILON, env = None, test_seed = SEED):
    setup_utils.setup_and_load(use_cmd_line_args=False, is_high_res=True, num_levels=NUM_LEVELS, set_seed=test_seed)
    
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
    print("result:", status, '\n')
    if RENDER_SCREEN and not IN_PYNB:
        env.render()
    return steps_done, max_reward



def gradeReplayMemory():
    print("Grading ReplayMemory...")
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


def gradeMakeBatch():
    print("Grading doMakeBatch...")
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

class GradeUnitTestDQN(nn.Module):
    def __init__(self, h, w, num_actions):
        super(GradeUnitTestDQN, self).__init__()
        self.num_actions = num_actions
    def forward(self, x):
        assert(False), "Network should not be queried when epsilon = 1.0." 
        return None

def gradeSelectAction():
    print("Grade select_action...")
    from scipy.stats import chisquare
    sample_size = 10000
    num_tests = 100
    pass_rate = 0.9
    screen_height = 40
    screen_width = 40
    epsilon = 1.0
    num_actions = 7
    test_results = {True: 0, False: 0}
    significance_level = 0.02
    net = GradeUnitTestDQN(screen_height, screen_width, num_actions).to(DEVICE)
    state = torch.randn(1, 3, 80, 80, device=DEVICE)
    for j in range(num_tests):
        samples = {}
        for i in range(sample_size):
            action, new_epsilon = select_action(state, net, num_actions, epsilon, steps_done = 0, bootstrap_threshold = 2)
            assert(type(action) == torch.Tensor and action.size() == (1,1)), "Action not correct shape."
            assert(new_epsilon == epsilon), "Epsilon should not change during bootstrapping."
            action = action.item()
            if action not in samples:
                samples[action] = 0
            samples[action] = samples[action] + 1
        expected = [sample_size / num_actions] * num_actions
        statistic, pvalue = chisquare(f_obs=list(samples.values()), f_exp=expected)
        test_results[pvalue >= significance_level] += 1
    assert(test_results[True] > pass_rate * num_tests), "Random sample is not from uniform distribution."    
    print("select_action test passed.")
    return True





def gradePredictQValues():
    print("Grading doPredictQValues...")
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

def gradePredictNextStateUtilities():
    print("Grading doPredictNextStateUtilities...")
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

def gradeComputeExpectedQValues():
    print("Grading doComputeExpectedQValues...")
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

def gradeComputeLoss():
    print("Grading doComputeLoss...")
    batch_size = 128
    state_action_values = torch.randn(batch_size, device=DEVICE)
    expected_state_action_values = torch.randn(batch_size, device=DEVICE)
    loss = doComputeLoss(state_action_values, expected_state_action_values)
    assert(type(loss) == torch.Tensor and len(loss.size()) == 0), "Loss not of expected shape."
    print("doComputeLoss test passed.")
    return True


def grade_unit_test():
    grade = 0
    try:
        gradeReplayMemory()
        grade = grade + 1
    except AssertionError:
        print("replay memory failed")
    try:
        gradeMakeBatch()
        grade = grade + 1
    except AssertionError:
        print("doMakeBatch failed")    
    try:
        gradeSelectAction()
        grade = grade + 1
    except AssertionError:
        print("select_action failed")
    try:
        gradePredictQValues()
        grade = grade + 1
    except AssertionError:
        print("doPredictQValues failed")
    try:
        gradePredictNextStateUtilities()
        grade = grade + 1
    except AssertionError:
        print("doPredictNextStateUtilities failed")
    try:
        gradeComputeExpectedQValues()
        grade = grade + 1
    except AssertionError:
        print("doComputeExpectedQValues failed")
    try:
        gradeComputeLoss()
        grade = grade + 1
    except AssertionError:
        print("doComputeLoss failed")
    return grade / 7.0

if not SKIP_UNIT_TESTS:
    print("#####")
    print("GRADING UNIT TESTS...")
    unit_test_grade = grade_unit_test()
    unit_test_grade = unit_test_grade * unit_test_weight
    print("unit test score:", unit_test_grade)

if not SKIP_TRAINING:
    print("#####")
    print("GRADING TRAINING...")
    #for _ in range(1):
    try:
        net = grade_train(num_episodes = 2,
                          load_filename = None,
                          save_filename = None, 
                          eval_interval = 2, 
                          replay_capacity = 2, 
                          bootstrap_threshold = 1, 
                          epsilon = EPSILON, 
                          eval_epsilon = EVAL_EPSILON, 
                          gamma = GAMMA, 
                          batch_size = MY_BATCH_SIZE, 
                          target_update = TARGET_UPDATE, 
                          random_seed = RANDOM_SEED, 
                          num_levels = NUM_LEVELS, 
                          seed = EASY_LEVEL)
        grad_pass = True
        for i, param in enumerate(net.parameters()):
            if param.grad is None:
                grad_pass = False
                print("GRADS ARE NONE")
                break
        evaluate(net, epsilon = EVAL_EPSILON, env = None, test_seed = EASY_LEVEL)
        if grad_pass:
            training_grade = 1.0

    except Exception as e: 
        print("TRAINING BROKE")
        training_grade = 0.0
        print(traceback.format_exc())

    training_grade = training_grade * training_weight
    print("training score:", training_grade)

    
print("#####")

best_easy_grade = 0.0
easy_grade = 0.0
best_medium_grade = 0.0
medium_grade = 0.0
best_monster_grade = 0.0
monster_grade = 0.0
best_easy_duration = None
best_medium_duration = None
best_monster_duration = None


if (is_all or is_easy) and os.path.exists('easy.model'):
    print("GRADING ON EASY LEVEL")
    easy_model = torch.load('easy.model')
    try:
        for i in range(10):
            print("EASY LEVEL TRY", i)
            easy_grade = 0.0
            total_duration = 0.0
            for j in range(10):
                duration, max_reward = grade_evaluate(easy_model, epsilon = 0.1, env = None, test_seed = EASY_LEVEL)
                status, score = episode_status(duration, max_reward)
                total_duration = total_duration + score
            average_duration = total_duration / 10.0
            if average_duration < 150:
                easy_grade = easy_grade + 3
            if average_duration < 100:
                easy_grade = easy_grade + 1
            if average_duration < 50:
                easy_grade = easy_grade + 1
            if easy_grade > best_easy_grade:
                best_easy_grade = easy_grade
            if best_easy_duration is None or average_duration < best_easy_duration:
                best_easy_duration = average_duration
            print("EASY SCORE", i, easy_grade, average_duration)
            if easy_grade >= 5.0:
                break
    except Exception as e: 
        print("EASY GRADING BROKE")
        print(traceback.format_exc())
    easy_model = None
    print("BEST EASY SCORE", best_easy_grade, best_easy_duration)

if (is_all or is_medium) and os.path.exists('medium.model'):
    print("GRADING ON MEDIUM LEVEL")
    medium_model = torch.load('medium.model')
    try:
        for i in range(10):
            print("MEDIUM LEVEL TRY", i)
            medium_grade = 0.0
            total_duration = 0.0
            for j in range(10):
                duration, max_reward = grade_evaluate(medium_model, epsilon = 0.1, env = None, test_seed = MEDIUM_LEVEL)
                status, score = episode_status(duration, max_reward)
                total_duration = total_duration + score
            average_duration = total_duration / 10.0
            if average_duration < 150:
                medium_grade = medium_grade + 1
            if medium_grade > best_medium_grade:
                best_medium_grade = medium_grade
            if best_medium_duration is None or average_duration < best_medium_duration:
                best_medium_duration = average_duration
            print("MEDIUM SCORE", i, medium_grade, average_duration)
            if medium_grade >= 1.0:
                break
    except Exception as e: 
        print("MEDIUM GRADING BROKE")
        print(traceback.format_exc())
    medium_model = None
    print("BEST MEDIUM SCORE", best_medium_grade, best_medium_duration)

if (is_all or is_monster) and os.path.exists('monster.model'):
    print("GRADING ON MONSTER LEVEL")
    monster_model = torch.load('monster.model')
    try:
        for i in range(10):
            print("MONSTER LEVEL TRY", i)
            monster_grade = 0.0
            total_duration = 0.0
            for j in range(10):
                duration, max_reward = grade_evaluate(monster_model, epsilon = 0.1, env = None, test_seed = ONE_MONSTER)
                status, score = episode_status(duration, max_reward)
                total_duration = total_duration + score
            average_duration = total_duration / 10.0
            if average_duration < 300:
                monster_grade = monster_grade + 1
            if monster_grade > best_monster_grade:
                best_monster_grade = monster_grade
            if best_monster_duration is None or average_duration < best_monster_duration:
                best_monster_duration = average_duration
            print("MONSTER SCORE", i, monster_grade, average_duration)
            if monster_grade >= 1.0:
                break
    except Exception as e: 
        print("MONSTER GRADING BROKE")
        print(traceback.format_exc())
    monster_model = None
    print("BEST MONSTER SCORE", best_monster_grade, best_monster_duration)

test_grade = best_easy_grade + best_medium_grade + best_monster_grade
test_grade = test_grade * test_weight

print("#####")
print("best easy duration:", best_easy_duration)
print("best medium duration", best_medium_duration)
print("best monster duration", best_monster_duration)

print("#####")
print("Unit test grade:", unit_test_grade)
print("Training grade:", training_grade)
print("Testing grade:" ,test_grade)

