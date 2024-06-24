from snake_environment import SnakeEnv
from snake_q_learning_agent import QLearningAgent

def main():
    env = SnakeEnv()
    agent = QLearningAgent(env)
    
    print("Training the agent...")
    agent.train(num_episodes=10000)
    
    print("\nWatching the trained agent play...")
    agent.play(num_games=5)

if __name__ == "__main__":
    main()