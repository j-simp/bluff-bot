import torch
from pettingzoo.classic import texas_holdem_no_limit_v6
from dqn_agent import DQNAgent
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


from evaluate import evaluate

# Used to train a new model
def train():
    
    num_episodes = 1000000
    adversary_update_interval = 100
    performance_log_interval = 1000
    best_reward = float('-inf')

    reward_log = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = texas_holdem_no_limit_v6.env(num_players=2)
    agent = DQNAgent(state_size=54, action_size=5, device=device)
    adversary = DQNAgent(state_size=54, action_size=5, device=device)

    agent_names = ["player_0", "player_1"]

    # Training loop
    for episode in tqdm(range(num_episodes)):
        curr_agent_name = np.random.choice(agent_names)
        env.reset()
        total_reward = 0
        done = False
        for agent_name in env.agent_iter():

            observation, reward, termination, truncation, _ = env.last()

            if termination or truncation:
                env.step(None)
                continue

            state = observation['observation']
            action_mask = observation['action_mask']
            if agent_name == curr_agent_name:
                action = agent.select_action(state, action_mask)
            else:
                action = adversary.act_deterministic(state, action_mask)
            env.step(action)
            next_observation, reward, termination, truncation, _ = env.last()
            next_state = next_observation['observation'] if next_observation else None
            done = termination or truncation
            agent.remember(state, action, next_state, reward, done)
            

            if agent_name == curr_agent_name:
                agent.optimize_model()

        if num_episodes % adversary_update_interval == 0:
            agent.save_model("adversary.pth")
            adversary.load_model("adversary.pth")

        if episode % performance_log_interval == 0:
            agent.save_model("dqn_model.pth")
            eval_reward = evaluate()
            reward_log.append(eval_reward)
            if eval_reward > best_reward:
                agent.save_model("best_model.pth")

    # Save model
    torch.save(agent.policy_net.state_dict(), "dqn_model.pth")
    
    # Graph training performance
    plt.plot([i for i in range(0, num_episodes, performance_log_interval)], reward_log)
    plt.xlabel("Evaluation Interval")
    plt.ylabel("Evaluation Reward")
    plt.title("Training Performance")
    plt.show()

if __name__ == "__main__":
    train()