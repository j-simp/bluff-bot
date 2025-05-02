import torch
from pettingzoo.classic import texas_holdem_no_limit_v6
import numpy as np
from dqn_agent import DQNAgent
from replay_buffer import ReplayMemory

def check_trained_model():
    # Load trained model and make sure all actions selected are valid
    env = texas_holdem_no_limit_v6.env(num_players=2)
    agent = DQNAgent(state_size, action_size, device)
    agent.policy_net.load_state_dict(torch.load("dqn_model.pth"))

    env.reset()
    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
        else:
            state = observation['observation']
            action_mask = observation['action_mask']
            action = agent.select_action(state, action_mask)
            valid_actions = np.nonzero(action_mask)
            assert np.isin(action, valid_actions).any(), "Invalid action chosen!"

        env.step(action)
    env.close()

    print("Passed check_trained_model() test!")

def check_replay_buffer():
    memory = ReplayMemory(200)
    env = texas_holdem_no_limit_v6.env(num_players=2)
    env.reset()
    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            env.step(None)
            continue
        else:
            state = observation['observation']
            action_mask = observation['action_mask']
            legal_actions = np.where(action_mask == 1)[0]
            action = np.random.choice(legal_actions)
            env.step(action)
            next_observation, reward, termination, truncation, _ = env.last()
            print(f"Next observation {next_observation}\n")
            next_state = next_observation['observation'] if next_observation else None
            print(f"Next state: {next_state}\n", f"Next state type: {type(next_state)}\n")
            done = termination or truncation
            memory.push(state, action, next_state, reward, done)

    env.close()

    x = memory.sample(1)
    print(x, "\n") 
    print(type(x[0].state))
    assert len(x[0].state) == 54, "States are not being saved to buffer correctly!"
    agent = DQNAgent(state_size, action_size, device)
    agent.policy_net.load_state_dict(torch.load("dqn_model.pth"))

    action = agent.act_deterministic(x[0].state)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_size = 54
    action_size = 5
    check_trained_model()
    check_replay_buffer()