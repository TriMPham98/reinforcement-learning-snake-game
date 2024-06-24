import numpy as np
from snake_environment import SnakeEnv

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = [self.get_q_value(state, a) for a in range(self.env.action_space.n)]
            return np.argmax(q_values)

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.get_q_value(state, action)
        next_max_q = max([self.get_q_value(next_state, a) for a in range(self.env.action_space.n)])
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[(state, action)] = new_q

    def train(self, num_episodes=10000):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = tuple(map(tuple, state))
            total_reward = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = tuple(map(tuple, next_state))
                self.update_q_value(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            if episode % 1000 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}")

    def play(self, num_games=5):
        for game in range(num_games):
            state, _ = self.env.reset()
            state = tuple(map(tuple, state))
            total_reward = 0
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = tuple(map(tuple, next_state))
                state = next_state
                total_reward += reward
                self.env.render()
                print(f"Action: {action}, Reward: {reward}")

            print(f"Game {game + 1} finished. Total Reward: {total_reward}")

# Usage
env = SnakeEnv()
agent = QLearningAgent(env)
agent.train()
agent.play()