import torch
import numpy as np
from pettingzoo.classic import texas_holdem_no_limit_v6
from dqn_agent import DQNAgent
from time import sleep

action_map = {0: "0 to Fold", 
              1: "1 to Check/Call", 
              2: "2 to Raise by Half the Pot Amount", 
              3: "3 to Raise by the Full Pot Amount", 
              4: "4 to GO ALL IN!"}

agent_names = ["player_0", "player_1"]

# Load the best DQN model
def load_agent(model_path, state_size, action_size, device):
    agent = DQNAgent(state_size=state_size, action_size=action_size, device=device)
    agent.policy_net.load_state_dict(torch.load(model_path))
    agent.policy_net.eval() 
    return agent

# Play against the model
def play_game():
    """
    Allows a human to play against a trained agent in the Texas Hold'em environment.
    """
    # Set up the environment and device
    env = texas_holdem_no_limit_v6.env(num_players=2, render_mode = 'human')
    env.reset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model_path = "best_model.pth"
    agent = load_agent(model_path, state_size=54, action_size=5, device=device)

    human_number = np.random.choice([1, 2])
    human_name = agent_names[human_number - 1]

    print(f"\n\nWelcome to Texas Hold'em! You are Player {human_number}.")
    print("Instructions will be displayed during your turn.\n")
    env.render()
    # Main game loop
    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            if agent_name == human_name:
                if reward > 0:
                    print(f"You win {reward * 2} chips!")
                elif reward < 0:
                    print(f"You lose {reward * 2} chips :(")
                else:
                    print("You didn't win or lose any chips.")
            env.step(None)
            continue

        if agent_name == human_name:
            print("\nYour Turn:\n")

            actions = np.where(observation['action_mask'] == 1)[0]
            print(f"Your available actions:")
            for key in actions:
                print(action_map[key])
            action = None
            while action not in np.where(observation['action_mask'] == 1)[0]:
                try:
                    action = int(input("Enter your action: "))
                except ValueError:
                    print("Invalid input. Please enter a valid action.")
            env.step(action)
        else:
            state = observation['observation']
            action_mask = observation['action_mask']
            action = agent.act_deterministic(state, action_mask)
            print(f"\nAI choses action: {action_map[action]}")
            env.step(action)

        # Render the environment in human mode

    # Close the environment after the game ends
    sleep(10)
    env.close()

if __name__ == "__main__":
    play_game()