from snake_environment import SnakeEnv
from snake_q_learning_agent import QLearningAgent
import time

def main():
    env = SnakeEnv()
    agent = QLearningAgent(env)
    
    while True:
        print("\nChoose an option:")
        print("1. Train the AI")
        print("2. Watch AI play")
        print("3. Play manually")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            print("Training the agent...")
            agent.train(num_episodes=10000)
        elif choice == '2':
            print("Watching the trained agent play...")
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
                    time.sleep(0.1)

                print(f"Game {game + 1} finished. Total Reward: {total_reward}")
        elif choice == '3':
            print("Starting manual play...")
            print("Use arrow keys to control the snake.")
            env.reset()
            env.manual_play()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

    env.close()

if __name__ == "__main__":
    main()