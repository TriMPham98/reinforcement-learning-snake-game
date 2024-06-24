from snake_environment import SnakeEnv
from snake_q_learning_agent import QLearningAgent
import time

def main():
    env = SnakeEnv()
    agent = QLearningAgent(env)
    
    print("Training the agent...")
    agent.train(num_episodes=10000)
    
    print("\nWatching the trained agent play...")
    for game in range(5):
        state, _ = env.reset()
        state = tuple(map(tuple, state))
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = tuple(map(tuple, next_state))
            state = next_state
            total_reward += reward
            env.render()
            time.sleep(0.1)  # Add a small delay to make the game visible

        print(f"Game {game + 1} finished. Total Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    main()