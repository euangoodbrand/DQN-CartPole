# %%
from utils import DQN, ReplayBuffer, greedy_action, epsilon_greedy, update_target, loss

import gym
from gym import Env

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import Adam, RAdam, SGD
import os
from collections import namedtuple
from collections import defaultdictSS
import matplotlib.patches as mpatches

# %% [markdown]
# ## Testing Hyperparameters against Convergence and Average Reward Metrics

# %%
# Define a named tuple to store transition details in the replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# Initialize the CartPole environment from the OpenAI Gym
env = gym.make('CartPole-v1')

def has_converged(rewards, threshold, window_size):
    """
    Checks if the learning has converged based on a moving average of rewards.

    Args:
    rewards (list): List of accumulated rewards per episode.
    threshold (float): The threshold value for the moving average to exceed for convergence.
    window_size (int): The number of episodes to consider for the moving average.

    Returns:
    bool: True if the average reward over the last `window_size` episodes exceeds `threshold`.
    """
    if len(rewards) >= window_size:
        return np.mean(rewards[-window_size:]) > threshold
    return False

# Define hyperparameters
NUM_RUNS = 10
NUM_EPISODES = 100
lr = [0.001, 0.01, 0.1]
buffer_range = [100,500,1000,5000,10000,20000]
e_range = [1,0.99,0.9,0]
e_decays = [0.1, 0.2,0.3,0.4, 0.5,0.6, 0.7 ,0.8,0.9 ,1]
policy_nn_range = [[4,32,16,2],[4,128,64,32,2],[4, 128, 64, 2],[4, 16, 2],[4, 32, 2],[4,64,2], [4, 128, 2]]
batch_sizes = [1, 5, 10, 20, 32]
convergence_threshold = 50
convergence_window = 10
optimisers = ['SGD','Adam','Radam','RMSprop']
act_funcs = [F.tanh, F.relu, F.leaky_relu, F.sigmoid]
iteration_range = [1,2,3,4,5,10,20]

# Initialize variables to track the best hyperparameter configuration
best_convergence_rate = float('inf')
best_average_reward = 0
best_hyperparameters = None

# Array to store results of different runs
runs_results = []

# Nested loops for hyperparameter tuning
for lr in lr:
    for buffer_size in buffer_range:
        for epsilon in e_range:
            for epsilon_decay in e_decays:
                for nn_architecture in policy_nn_range:
                    for batch_size in batch_sizes:
                        config_results = []

                        for run in range(NUM_RUNS):
                            # Initialize the policy network, target network, optimizer, and replay buffer for each run
                            policy_net = DQN(nn_architecture, F.relu)
                            target_net = DQN(nn_architecture, F.relu)
                            update_target(target_net, policy_net)
                            optimizer = optim.Adam(policy_net.parameters(), lr=lr)
                            memory = ReplayBuffer(buffer_size)

                            steps_done = epsilon
                            episode_rewards = []
                            converged_in_episode = None

                            for i_episode in range(NUM_EPISODES):
                                # Environment reset and initial state preparation
                                observation, info = env.reset()
                                state = torch.tensor(observation).float()
                                current_episode_reward = 0
                                done = False

                                while not done:
                                    # Select and perform an action using epsilon-greedy policy
                                    action = epsilon_greedy(steps_done, policy_net, state)
                                    response = env.step(action)

                                    # Handling different formats of responses from the environment
                                    if len(response) == 5:
                                        next_observation, reward, done, extra_boolean, info = response
                                    else:
                                        raise ValueError(f"Unexpected response format from env.step: {response}")

                                    # Update state and store the transition in the replay buffer
                                    next_state = torch.tensor(next_observation).float()
                                    reward_tensor = torch.tensor([reward])
                                    memory.push((state, torch.tensor([action]), next_state, reward_tensor, torch.tensor([done])))

                                    state = next_state
                                    current_episode_reward += reward

                                    # Training the model with a batch of transitions from the replay buffer
                                    if len(memory.buffer) >= batch_size:
                                        transitions = memory.sample(batch_size)
                                        batch = Transition(*zip(*transitions))
                                        # Extract and reshape tensors from the batch
                                        state_batch = torch.cat(batch.state).view(batch_size, -1)
                                        action_batch = torch.cat(batch.action).view(batch_size, -1)
                                        reward_batch = torch.cat(batch.reward).view(batch_size, -1)
                                        next_state_batch = torch.cat(batch.next_state).view(batch_size, -1)
                                        done_batch = torch.cat(batch.done).view(batch_size, -1)

                                        mse_loss = loss(policy_net, target_net, state_batch, action_batch, reward_batch, next_state_batch, done_batch)
                                        optimizer.zero_grad()
                                        mse_loss.backward()
                                        optimizer.step()

                                    # Decay epsilon after each action
                                    steps_done *= epsilon_decay

                                episode_rewards.append(current_episode_reward)
                                if converged_in_episode is None and has_converged(episode_rewards, convergence_threshold, convergence_window):
                                    converged_in_episode = i_episode

                            config_results.append(episode_rewards)
                        
                            runs_results.append(config_results)

                            # Update best hyperparameters if current configuration is better
                            if converged_in_episode is not None and len(episode_rewards) > converged_in_episode:
                                average_reward_post_convergence = np.mean(episode_rewards[converged_in_episode:])
                                convergence_rate = converged_in_episode

                                if convergence_rate < best_convergence_rate or (convergence_rate == best_convergence_rate and average_reward_post_convergence > best_average_reward):
                                    best_convergence_rate = convergence_rate
                                    best_average_reward = average_reward_post_convergence
                                    best_hyperparameters = {
                                        'lr': lr,
                                        'buffer_size': buffer_size,
                                        'epsilon': epsilon,
                                        'epsilon_decay': epsilon_decay,
                                        'nn_architecture': nn_architecture,
                                        'batch_size': batch_size
                                    }        
                        env.close()

print("Best Hyperparameters based on Convergence Rate and Average Reward Post-Convergence:")
print(best_hyperparameters)


# %%
def select_optimizer(policy_network, lr, optimiser_name):
    """
    Selects and initializes an optimizer for the policy network based on the given name.

    Args:
    policy_network (nn.Module): The neural network model for which the optimizer will be used.
    lr (float): Learning rate for the optimizer.
    optimiser_name (str): Name of the optimizer to be used.

    Returns:
    torch.optim.Optimizer: An initialized optimizer.

    Raises:
    ValueError: If the optimizer name is not recognized.
    """
    optimizers = {
        'Adam': optim.Adam,
        'RAdam': optim.RAdam,
        'SGD': optim.SGD,
        'RMSprop' : optim.RMSprop 
    }
    if optimiser_name in optimizers:
        return optimizers[optimiser_name](policy_network.parameters(), lr=lr)
    else:
        raise ValueError("Invalid optimizer")

def initialize_environment():
    """
    Initializes and returns the CartPole-v1 environment from OpenAI Gym.

    Returns:
    gym.Env: The initialized environment.
    """
    return gym.make('CartPole-v1')

def initialize_episode(env):
    """
    Initializes a new episode in the given environment.

    Args:
    env (gym.Env): The environment in which the episode is to be initialized.

    Returns:
    torch.Tensor: The initial state of the environment as a tensor.
    """
    observation, _ = env.reset()
    state = torch.tensor(observation).float()
    return state

def perform_episode_step(env, policy_network, state, memory, e):
    """
    Performs a single step in an episode, including action selection, environment interaction, and memory update.

    Args:
    env (gym.Env): The environment in which the episode is occurring.
    policy_network (nn.Module): The policy network used for action selection.
    state (torch.Tensor): The current state of the environment.
    memory (ReplayBuffer): The memory buffer to store experiences.
    e (float): The epsilon value for epsilon-greedy action selection.

    Returns:
    tuple: The next state, and boolean flags indicating if the episode is done or terminated.
    """
    action = epsilon_greedy(e, policy_network, state)
    observation, reward, done, terminated, _ = env.step(action)
    reward = torch.tensor([reward])
    action = torch.tensor([action])
    next_state = torch.tensor(observation).reshape(-1).float()
    memory.push([state, action, next_state, reward, torch.tensor([done])])
    return next_state, done, terminated

def batch(policy_network, target_net, optimizer, memory, batch_space):
    """
    Processes a batch of experiences to train the policy network.

    Args:
    policy_network (nn.Module): The policy network to be trained.
    target_net (nn.Module): The target network used for stable Q-value estimation.
    optimizer (torch.optim.Optimizer): The optimizer used for training.
    memory (ReplayBuffer): The replay buffer containing experiences.
    batch_space (int): The size of the batch to be used for training.
    """
    if len(memory.buffer) < batch_space:
        return

    transitions = memory.sample(batch_space)
    state_batch, action_batch, nextstate_batch, reward_batch, dones = (torch.stack(x) for x in zip(*transitions))
    mse_loss = loss(policy_network, target_net, state_batch, action_batch, reward_batch, nextstate_batch, dones)
    optimizer.zero_grad()
    mse_loss.backward()
    optimizer.step()

def single_run(env, policy_network, target_net, optimizer, memory, episodes, batch_space, e, e_decay, update_target_iteration):
    """
    Executes a single run of the training process.

    Args:
    env (gym.Env): The environment to train in.
    policy_network (nn.Module): The policy network.
    target_net (nn.Module): The target network.
    optimizer (torch.optim.Optimizer): The optimizer for training.
    memory (ReplayBuffer): The replay buffer.
    episodes (int): The number of episodes to run.
    batch_space (int): The batch size for training.
    e (float): Initial epsilon value for epsilon-greedy strategy.
    e_decay (float): Decay rate for epsilon.
    update_target_iteration (int): Number of episodes after which the target network is updated.

    Returns:
    list: A list of episode durations for the run.
    """
    episode_durations = []

    for i_episode in range(episodes):
        state = initialize_episode(env)
        done = False
        terminated = False
        t = 0

        while not (done or terminated):
            state, done, terminated = perform_episode_step(env, policy_network, state, memory, e)
            batch(policy_network, target_net, optimizer, memory, batch_space)
            
            if done or terminated:
                episode_durations.append(t + 1)
            t += 1

            e = max(0.01, e * e_decay)

        if i_episode % update_target_iteration == 0: 
            update_target(target_net, policy_network)

    return episode_durations

def run_dqn(e, e_decay, lr, optimiser, policy_nn, act_func, batch_space, memory_replay_buffer_size, update_target_iteration, episodes, num_runs=10):
    """
    Executes the DQN training process over a specified number of runs.

    Args:
    e (float): Initial epsilon value for epsilon-greedy strategy.
    e_decay (float): Decay rate for epsilon.
    lr (float): Learning rate for the optimizer.
    optimiser (str): Name of the optimizer to be used.
    policy_nn (list): Neural network architecture for the policy network.
    act_func (function): Activation function to be used in the policy network.
    batch_space (int): The batch size for training.
    memory_replay_buffer_size (int): The size of the replay buffer.
    update_target_iteration (int): Number of episodes after which the target network is updated.
    episodes (int): The number of episodes per run.
    num_runs (int, optional): The number of training runs to perform. Defaults to 1.

    Returns:
    list: A list containing the episode durations for each run.
    """
    runs_results = []
    env = initialize_environment()

    for _ in range(num_runs):
        policy_network = DQN(policy_nn, act_func)
        target_net = DQN(policy_nn, act_func)
        update_target(target_net, policy_network)
        target_net.eval()

        optimizer = select_optimizer(policy_network, lr, optimiser)

        memory = ReplayBuffer(memory_replay_buffer_size)

        episode_durations = single_run(
            env, policy_network, target_net, optimizer, memory, episodes, batch_space, e, e_decay, update_target_iteration
        )
        runs_results.append(episode_durations)

    return runs_results


# Define hyperparameters as individual variables
# Define hyperparameters for the DQN algorithm
e = 0.8  # Initial epsilon value for epsilon-greedy action selection
e_decay = 0.99  # Epsilon decay rate after each episode
update_target_iteration = 1  # Number of episodes after which to update the target network
lr = 0.001  # Learning rate for the optimizer
act_func = F.leaky_relu  # Activation function to be used in the neural network
batch_space = 32  # Batch size for training
memory_replay_buffer_size = 2000  # Size of the replay buffer
optimiser = 'Adam'  # Name of the optimizer to be used
policy_nn = [4, 128, 64, 2]  # Neural network architecture for the policy network

episodes = 300  # Number of episodes to run the DQN algorithm

# Run the DQN algorithm with the defined hyperparameters and print the results
# runs_results = run_dqn(
#     e, e_decay, lr, optimiser, policy_nn, act_func,
#     batch_space, memory_replay_buffer_size, update_target_iteration, episodes
# )

# print(runs_results)


# %%
results_tensor = torch.tensor(runs_results)

means = results_tensor.float().mean(dim=0)
stds = results_tensor.float().std(dim=0)

episodes_value = 300
# Plotting
episodes = torch.arange(1, episodes_value + 1)  # Adjusted to match the number of episodes
plt.plot(episodes, means, label='Mean Duration')
plt.fill_between(episodes, means - stds, means + stds, alpha=0.3, color='blue', label='Std. Deviation')

plt.axhline(y=100, color='red', linestyle='--', label='Target Duration')
plt.title('Training Performance Over Episodes')
plt.ylabel("Average Episode Duration")
plt.xlabel("Episode")
plt.legend()
plt.show()

# %%
# Constants
NUM_EPISODES = 250
EPISODE_RANGE = torch.arange(NUM_EPISODES)
BLUE_LINE_Y = 100
BLUE_LINE_COLOR = 'blue'
BLUE_LINE_STYLE = '--'
FIG_SIZE = (20, 10)
FONT_SIZE = 18
LEGEND_LOC = 'upper left'
LINE_WIDTH = 2
GRID_STYLE = 'dashed'
COLOR_PALETTE = plt.cm.viridis  # Using a color palette for lines

def parameter_testing(baseline, parameter, values):
    results = []
    print(f"Testing parameter: {parameter}")

    for value in values:
        configuration = {**baseline, parameter: value}
        print(f"Parameter value: {value}\n{configuration}")

        run_results = run_dqn(num_runs=1, **configuration)
        results.append(run_results)

    return results

def param_plot(range_of_results, parameters, title, save_path='images'):
    # Check if the save directory exists, create if not
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.style.use('seaborn-darkgrid')  # Enhanced style
    plt.figure(figsize=FIG_SIZE)
    plt.rcParams.update({'font.size': FONT_SIZE})

    num_values = len(parameters)
    for i, (parameter_value, runs_results) in enumerate(zip(parameters, range_of_results)):
        means = torch.tensor(runs_results).float().mean(0)
        plt.plot(EPISODE_RANGE, means, label=str(parameter_value), linewidth=LINE_WIDTH,
                 color=COLOR_PALETTE(i / num_values))  # Apply color from palette

    plt.ylabel("Average Return", fontweight='bold')
    plt.xlabel("Episode", fontweight='bold')
    plt.title(title, fontweight='bold')
    plt.axhline(y=BLUE_LINE_Y, color=BLUE_LINE_COLOR, linestyle=BLUE_LINE_STYLE, linewidth=LINE_WIDTH)
    plt.legend(loc=LEGEND_LOC)
    plt.grid(True, linestyle=GRID_STYLE)

    # Save the figure
    plt.savefig(os.path.join(save_path, title.replace(" ", "_") + '.png'))

    plt.show()


# Hyperparemeters used when testing other parameters in isolation.
hyperparams = {
    "lr": 0.001,
    "e": 0.8,
    "memory_replay_buffer_size": 20000,
    "e_decay": 0.99,
    "optimiser": 'Adam',
    "act_func": F.leaky_relu,
    "policy_nn": [4, 128, 64, 2],
    "batch_space": 20,
    "update_target_iteration": 1,
    "episodes": NUM_EPISODES
}


e_range = [0.1,0.3,0.5,0.7,0.9]

e_decay_range = [0,0.9,0.99,1]

activation_functions = [F.tanh, F.relu, F.leaky_relu, F.sigmoid]

batch_range = [1,5,10,20,32]

buffer_range = [100,500,1000,5000,10000,20000]

lr_range = [0.001,0.01,0.1,0.5]

optimiser_range = ['SGD', 'Adam', 'RAdam', 'RMSprop']

policy_neural_networks = [
        [4, 16, 2], 
        [4, 32, 2], 
        [4, 128, 2], 
        [4, 32, 16, 2], 
        [4, 128, 64, 2],  
        [4, 64, 32, 16, 2]]

iteration_range = [1,2,3,4,5,10,20]


epsilon_results = parameter_testing(hyperparams, 'e', e_range)
param_plot(epsilon_results, e_range, "Epsilon Parameters Values")

exploration_results = parameter_testing(hyperparams, 'e', e_range)
param_plot(exploration_results, e_range, "Epsilon Parameters Values")

e_decay_results = parameter_testing(hyperparams, 'e_decay', e_decay_range)
param_plot(e_decay_results, e_decay_range, "Epsilon Decay Parameter Values")

activation_func_results = parameter_testing(hyperparams, 'act_func', activation_functions)
param_plot(activation_func_results, [f.__name__ for f in activation_functions], "Activation Function Parameter Values")

batch_results = parameter_testing(hyperparams, 'batch_space', batch_range)
param_plot(batch_results, batch_range, "Batch Size Parameter Values")

buffer_results = parameter_testing(hyperparams, 'memory_replay_buffer_size', buffer_range)
param_plot(buffer_results, buffer_range, "Memory Replay Buffer Size Parameter Values")

lr_results = parameter_testing(hyperparams, 'lr', lr_range)
param_plot(lr_results, lr_range, "Learning Rate Parameter Values")

optimiser_results = parameter_testing(hyperparams, 'optimiser', optimiser_range)
param_plot(optimiser_results, optimiser_range, "Optimizer Type Parameter Values")

policy_nn_results = parameter_testing(hyperparams, 'policy_nn', policy_neural_networks)
param_plot(policy_nn_results, [str(nn) for nn in policy_neural_networks], "Policy Neural Network Structure Parameter Values")

iteration_results = parameter_testing(hyperparams, 'update_target_iteration', iteration_range)
param_plot(iteration_results, iteration_range, "Update Target Iteration Parameter Values")

# %%

# Define hyperparameters as individual variables
e = 0.8
e_decay = 0.99
lr = 0.001
optimiser = 'Adam'
policy_nn = [4, 128, 64, 2]
act_func = F.leaky_relu
batch_space = 10
memory_replay_buffer_size = 2000
update_target_iteration = 1
episodes = 300

# Execute the training
learning_curve = run_dqn(
    e, e_decay, lr, optimiser, policy_nn, act_func,
    batch_space, memory_replay_buffer_size, update_target_iteration, episodes
)


# Define hyperparameters as individual variables
e = 1
e_decay = 1
lr = 1
optimiser = 'Adam'
policy_nn = [4,2]
act_func = F.relu
batch_space = 1
memory_replay_buffer_size = 1
update_target_iteration = 1
episodes = 300

# Execute the training
original = run_dqn(
    e, e_decay, lr, optimiser, policy_nn, act_func,
    batch_space, memory_replay_buffer_size, update_target_iteration, episodes
)


# %%
plt.figure(figsize=(20, 10))
plt.rcParams.update({'font.size': 22, 'font.family': 'serif'})

res = torch.tensor(original)
m = res.float().mean(0)
s = res.float().std(0)

returns = torch.tensor(learning_curve)
means = returns.float().mean(0)
std = returns.float().std(0)

plt.plot(torch.arange(300), m, label="Original hyperparameters", linestyle='--', color='green')
plt.fill_between(np.arange(300), m, m + s, alpha=0.5, color='green', hatch='.')
plt.fill_between(np.arange(300), m, m - s, alpha=0.5, color='green', hatch='.')

plt.plot(torch.arange(300), means, label="Our final tuned hyperparameters", color='purple')
plt.fill_between(np.arange(300), means, means + std, alpha=0.5, color='purple', hatch='\\')
plt.fill_between(np.arange(300), means, means - std, alpha=0.5, color='purple', hatch='\\')

plt.ylabel("Return")
plt.xlabel("Episode")
plt.title("Learning Curve of Model with Optimised Parameters")
plt.axhline(y=100, color='r', linestyle='--')
plt.legend(loc='upper left', frameon=True, framealpha=0.7)

plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().set_facecolor('#f2f2f2')

plt.show()

# %%
#|This function selects an optimizer for the neural network based on the specified type.
# Parameters:
# - net: The neural network for which the optimizer is being chosen.
# - lr_rate: Learning rate for the optimizer.
# - opt_type: The type of optimizer to use. Options are 'Adam', 'RAdam', 'SGD', and 'RMSprop'.
# Returns: An instantiated optimizer object for the given neural network.
def choose_optimizer(net, lr_rate, opt_type):
    # Dictionary mapping optimizer type names to their corresponding PyTorch classes.
    opt_choices = {
        'Adam': optim.Adam,
        'RAdam': optim.RAdam,
        'SGD': optim.SGD,
        'RMSprop': optim.RMSprop 
    }

    # If the chosen optimizer type is in the available choices, create and return the optimizer.
    if opt_type in opt_choices:
        return opt_choices[opt_type](net.parameters(), lr=lr_rate)
    else:
        # If an invalid optimizer type is passed, raise an error.
        raise ValueError("Invalid optimizer type")

# This function sets up the training environment using OpenAI Gym.
# Returns: The created Gym environment instance, specifically the 'CartPole-v1' environment.
def setup_env():
    return gym.make('CartPole-v1')

# This function starts a new round in the game environment.
# Parameters:
# - env: The Gym environment in which the round is to be started.
# Returns: The initial state vector of the environment after reset.
def begin_round(env):
    # Reset the environment and get the initial observation.
    obs, _ = env.reset()

    # Convert the observation to a PyTorch tensor and return it as the state vector.
    state_vec = torch.tensor(obs).float()
    return state_vec

# This function performs an action in the game environment based on the current state.
# Parameters:
# - env: The Gym environment in which the action is to be performed.
# - net: The neural network used for decision making.
# - present_state: The current state of the environment.
# - memory_buffer: The memory buffer to store experiences for training.
# - eps: Epsilon value for epsilon-greedy strategy in decision making.
# Returns: The next state after the action, and flags indicating whether the round is complete or finished.
def perform_action(env, net, present_state, memory_buffer, eps):
    # Determine the action to take using epsilon-greedy strategy.
    action_chosen = epsilon_greedy(eps, net, present_state)

    # Perform the chosen action in the environment and get the new observation and other info.
    obs, reward, complete, finished, _ = env.step(action_chosen)

    # Convert reward and action chosen to PyTorch tensors.
    reward_val = torch.tensor([reward])
    action_val = torch.tensor([action_chosen])

    # Convert the new observation to a tensor and reshape it as needed.
    next_state = torch.tensor(obs).reshape(-1).float()

    # Store the experience in the memory buffer.
    memory_buffer.push([present_state, action_val, next_state, reward_val, torch.tensor([complete])])

    # Return the next state and flags indicating round completion status.
    return next_state, complete, finished

# Process a batch for training
def process_batch(net, target_net, opt, memory_buffer, size_batch):
    """
    Processes a batch of data for training the network.

    Args:
    - net: The current neural network model being trained.
    - target_net: The target network used for calculating the loss.
    - opt: The optimizer used for updating the weights.
    - memory_buffer: The replay buffer containing the experiences.
    - size_batch: The size of the batch to be processed.

    This function will sample a batch from the memory buffer and
    perform a training step, including forward and backward propagation.
    """
    # Check if enough samples are available in the memory buffer
    if len(memory_buffer.buffer) < size_batch:
        return

    # Sample a batch of experiences from the memory buffer
    batch = memory_buffer.sample(size_batch)
    # Unpack the batch into respective components and stack them
    states, actions, next_states, rewards, completion = (torch.stack(x) for x in zip(*batch))
    # Calculate the loss
    loss_val = loss(net, target_net, states, actions, rewards, next_states, completion)
    # Reset gradients before backpropagation
    opt.zero_grad()
    # Backward pass to compute gradients
    loss_val.backward()
    # Update the weights
    opt.step()


# Conduct a single training session
def conduct_training(env, net, target_net, opt, memory_buffer, episodes_total, size_batch, eps, decay_eps, freq_update_target):
    """
    Conducts a training session over a given number of episodes.

    Args:
    - env: The environment for the agent to interact with.
    - net: The current neural network model.
    - target_net: The target network for stable Q-value estimation.
    - opt: The optimizer for updating the network weights.
    - memory_buffer: The replay buffer for storing experiences.
    - episodes_total: Total number of episodes for training.
    - size_batch: Batch size for training.
    - eps: Epsilon value for epsilon-greedy policy.
    - decay_eps: Decay rate for epsilon.
    - freq_update_target: Frequency of updating the target network.

    Returns:
    - lengths_episodes: A list containing the lengths of each episode.

    This function conducts training over a specified number of episodes,
    updating the network weights and target network at specified intervals.
    """
    lengths_episodes = []

    for ep in range(episodes_total):
        present_state = begin_round(env)
        complete = False
        finished = False
        count_steps = 0

        while not (complete or finished):
            # Perform an action based on the current state
            present_state, complete, finished = perform_action(env, net, present_state, memory_buffer, eps)
            # Process the batch for training
            process_batch(net, target_net, opt, memory_buffer, size_batch)
            
            # Log episode length
            if complete or finished:
                lengths_episodes.append(count_steps + 1)
            count_steps += 1

            # Update epsilon using decay
            eps = max(0.01, eps * decay_eps)

        # Update target network at specified intervals
        if ep % freq_update_target == 0: 
            update_target(target_net, net)

    return lengths_episodes


# Implement the DQN strategy
def implement_dqn_strategy(eps, decay_eps, rate_lr, opt_type, arch_net, func_activation, size_batch, size_memory, freq_update, episodes_total, count_runs=1):
    """
    Implements the Deep Q-Network (DQN) strategy for a given environment.

    Args:
    - eps: Starting value of epsilon for the epsilon-greedy policy.
    - decay_eps: Decay rate for epsilon.
    - rate_lr: Learning rate for the optimizer.
    - opt_type: Type of optimizer to use.
    - decay_wt: Weight decay for the optimizer.
    - arch_net: Architecture of the neural network.
    - func_activation: Activation function for the neural network.
    - size_batch: Batch size for training.
    - size_memory: Size of the replay buffer.
    - freq_update: Frequency of updating the target network.
    - episodes_total: Total number of training episodes.
    - count_runs: Number of runs to perform (default is 1).

    Returns:
    - net: The trained neural network model.

    This function sets up the environment and DQN network, then conducts
    training over a number of runs, returning the trained network.
    """
    outcomes = []
    env = setup_env()

    for _ in range(count_runs):
        # Initialize the DQN network and the target network
        net = DQN(arch_net, func_activation)
        target_net = DQN(arch_net, func_activation)
        update_target(target_net, net)
        target_net.eval()

        # Choose the optimizer based on input parameters
        opt = choose_optimizer(net, rate_lr, opt_type)

        # Initialize the replay buffer
        memory_buffer = ReplayBuffer(size_memory)

        # Conduct the training session
        durations_episodes = conduct_training(
            env, net, target_net, opt, memory_buffer, episodes_total, size_batch, eps, decay_eps, freq_update
        )
        outcomes.append(durations_episodes)

    return net


# Model configuration
model_config = {
    "eps": 0.8,
    "decay_eps": 0.99,
    "rate_lr": 0.001,
    "opt_type": 'Adam',
    "arch_net": [4, 128, 64, 2],
    "func_activation": F.leaky_relu,
    "size_batch": 10,
    "size_memory": 20000,
    "freq_update": 1,
    "episodes_total": 300,
    "count_runs": 10
}

# Execute the refactored function
trained_dqn = implement_dqn_strategy(**model_config)


# %%
def prepare_data(dqn_policy, cart_velocity, compute_value):
    """
    Prepares the data for plotting by computing the values based on the provided function 'compute_value'.
    """
    max_cart_angle = .2095
    max_angular_velocity = 2

    num_angle_samples = 100
    num_omega_samples = 100
    cart_angles = torch.linspace(max_cart_angle, -max_cart_angle, num_angle_samples)
    angular_velocities = torch.linspace(-max_angular_velocity, max_angular_velocity, num_omega_samples)

    value_matrix = torch.zeros((num_angle_samples, num_omega_samples))
    for angle_idx, cart_angle in enumerate(cart_angles):
        for omega_idx, angular_velocity in enumerate(angular_velocities):
            cart_state = torch.tensor([0., cart_velocity, cart_angle, angular_velocity])
            with torch.no_grad():
                action_values = dqn_policy(cart_state)
                value_matrix[angle_idx, omega_idx] = compute_value(action_values)
    
    return cart_angles, angular_velocities, value_matrix

def plot_matrix(cart_angles, angular_velocities, value_matrix, plot_title, cmap, colorbar=False, save_path=''):
    """
    Plots the provided matrix using matplotlib and saves it to a file.
    """
    plt.figure(figsize=(10,8))
    plt.rcParams.update({'font.size': 14})

    plt.contourf(cart_angles, angular_velocities, value_matrix.T, cmap=cmap, levels=100 if colorbar else None)
    if colorbar:
        cbar = plt.colorbar()
        cbar.set_label('Q-value', rotation=0, labelpad=15)  # Label for the color bar
    if not colorbar:
        # Updated legend descriptions
        right_action_patch = plt.matplotlib.patches.Patch(color='yellow', label='Push cart to right')
        left_action_patch = plt.matplotlib.patches.Patch(color='blue', label='Push cart to left')
        plt.legend(handles=[right_action_patch, left_action_patch])
    
    plt.xlabel("Angle (rad)")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.title(plot_title)

    if save_path:
        if not os.path.exists('images'):
            os.makedirs('images')
        plt.savefig(f'images/{save_path}.png', bbox_inches='tight')
    plt.show()

def plot_policy_actions(dqn_policy, cart_velocity, plot_title, file_name):
    cart_angles, angular_velocities, action_decision_matrix = prepare_data(
        dqn_policy, cart_velocity, lambda values: values.argmax()
    )
    plot_matrix(cart_angles, angular_velocities, action_decision_matrix, plot_title, cmap='cividis', save_path=file_name)

def plot_q_value_visualization(dqn_policy, cart_velocity, plot_title, file_name):
    cart_angles, angular_velocities, q_value_matrix = prepare_data(
        dqn_policy, cart_velocity, lambda values: values.max()
    )
    plot_matrix(cart_angles, angular_velocities, q_value_matrix, plot_title, cmap='cividis', colorbar=True, save_path=file_name)

# trained_dqn = runs_results

# Policy Actions Plots
plot_policy_actions(trained_dqn, cart_velocity=0, plot_title="Greedy Policy: Position 0, Velocity 0", file_name="policy_action_0_0")
plot_policy_actions(trained_dqn, cart_velocity=0.5, plot_title="Greedy Policy: Position 0, Velocity 0.5", file_name="policy_action_0_0.5")
plot_policy_actions(trained_dqn, cart_velocity=1, plot_title="Greedy Policy: Position 0, Velocity 1", file_name="policy_action_0_1")
plot_policy_actions(trained_dqn, cart_velocity=2, plot_title="Greedy Policy: Position 0, Velocity 2", file_name="policy_action_0_2")

# Q Value Visualization Plots
plot_q_value_visualization(trained_dqn, cart_velocity=0, plot_title="Q Position: Position 0, Velocity 0", file_name="q_value_vis_0_0")
plot_q_value_visualization(trained_dqn, cart_velocity=0.5, plot_title="Q Position: Position 0, Velocity 0.5", file_name="q_value_vis_0_0.5")
plot_q_value_visualization(trained_dqn, cart_velocity=1, plot_title="Q Position: Position 0, Velocity 1", file_name="q_value_vis_0_1")
plot_q_value_visualization(trained_dqn, cart_velocity=2, plot_title="Q Position: Position 0, Velocity 2", file_name="q_value_vis_0_2")


