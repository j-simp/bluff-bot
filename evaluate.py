# Not fully implemented. Need to change input and output dimensions
# of variables and change how environment fields are accessed

import torch
from pettingzoo.classic import texas_holdem_no_limit_v6
from dqn_agent import DQNAgent
import numpy as np

def evaluate(train=True):
    # Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = texas_holdem_no_limit_v6.env(num_players=2)
    state_size = 54
    action_size = 5
    agent = DQNAgent(state_size, action_size, device)
    agent.policy_net.load_state_dict(torch.load("best_model.pth"))

    agent_names = ["player_0", "player_1"]
    
    # Evaluation
    num_episodes = 10000
    total_reward = 0
    for episode in range(num_episodes):
        train_agent_id = np.random.choice(agent_names)
        env.reset()
        for agent_name in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                env.step(None)
                continue

            if agent_name == train_agent_id:
                state = observation['observation']
                action_mask = observation['action_mask']
                action = agent.act_deterministic(state, action_mask)
                env.step(action)
                _, reward, _, _, _ = env.last()
                total_reward += reward
                
            else:
                action_mask = observation['action_mask']
                env.step(env.action_space(agent_name).sample(action_mask))
    if not train:
        print("========================")
        print(f"Average Reward: {total_reward / num_episodes}")
        print("========================")
    return (total_reward / num_episodes)
    

if __name__ == "__main__":
    evaluate(train=False)
