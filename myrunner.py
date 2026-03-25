from mycheckersenv import env
from myagent import ACAgent
import numpy as np


def train(episodes=50):
    e = env()

    state_size = 37
    action_size = 144

    agent = ACAgent(state_size, action_size)

    rewards_history = []

    for ep in range(episodes):
        print(f"\nEpisode {ep} started")

        e.reset()
        total_reward = 0
        step_count = 0

        for agent_name in e.agent_iter():
            state, reward, term, trunc, _ = e.last()

            step_count += 1
            if step_count > 200:
                print("step limit reached")
                break

            if term or trunc:
                e.step(None)
                continue

            player = e.agent_name_mapping[agent_name]
            valid_moves = e._get_valid_moves(player)
            valid_actions = [(r * 6 + c) * 4 + d for (r, c, d) in valid_moves]

            # use model, but fallback to valid move
            action, log_prob = agent.select_action(state)
            if action not in valid_actions:
                action = np.random.choice(valid_actions)

            e.step(action)

            next_state, next_reward, next_term, next_trunc, _ = e.last()
            done = next_term or next_trunc

            agent.update(state, log_prob, reward, next_state, done)

            total_reward += reward

        rewards_history.append(total_reward)

        print(f"Episode {ep} finished | Reward: {total_reward}")

    return agent, rewards_history

def play_game(agent):
    e = env()
    e.reset()

    print("\n===== SAMPLE GAME =====")

    step_count = 0

    for agent_name in e.agent_iter():
        state, reward, term, trunc, _ = e.last()

        e.render()

        step_count += 1
        if step_count > 50:
            print("stopping display early")
            break

        if term or trunc:
            e.step(None)
            continue

        player = e.agent_name_mapping[agent_name]
        valid_moves = e._get_valid_moves(player)
        valid_actions = [(r * 6 + c) * 4 + d for (r, c, d) in valid_moves]

        action, _ = agent.select_action(state)
        if action not in valid_actions:
            action = np.random.choice(valid_actions)

        e.step(action)

    print("\n===== GAME END =====")


if __name__ == "__main__":
    trained_agent, rewards = train(50)

    print("\nFinal cumulative reward:", rewards[-1])

    play_game(trained_agent)