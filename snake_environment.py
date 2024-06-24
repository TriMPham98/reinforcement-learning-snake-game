import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SnakeEnv(gym.Env):
    def __init__(self, grid_size=10):
        super(SnakeEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # 0: up, 1: right, 2: down, 3: left
        self.observation_space = spaces.Box(low=0, high=2, shape=(grid_size, grid_size), dtype=np.uint8)
        
        self.snake = [(grid_size // 2, grid_size // 2)]
        self.food = None
        self.direction = 1
        self.steps = 0
        self.max_steps = 100 * grid_size

    def reset(self):
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.place_food()
        self.direction = 1
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        self._move_snake(action)
        done = self._is_collision() or self.steps >= self.max_steps
        reward = 0
        if self.snake[0] == self.food:
            reward = 1
            self.place_food()
        elif done:
            reward = -1
        return self._get_obs(), reward, done, False, {}

    def _move_snake(self, action):
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        head_x, head_y = self.snake[0]
        new_head = ((head_x + dx) % self.grid_size, (head_y + dy) % self.grid_size)
        if new_head == self.food:
            self.snake = [new_head] + self.snake
        else:
            self.snake = [new_head] + self.snake[:-1]

    def _is_collision(self):
        return self.snake[0] in self.snake[1:]

    def place_food(self):
        while True:
            self.food = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            if self.food not in self.snake:
                break

    def _get_obs(self):
        obs = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        for x, y in self.snake:
            obs[y, x] = 1
        fx, fy = self.food
        obs[fy, fx] = 2
        return obs

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:] = '.'
        for x, y in self.snake:
            grid[y, x] = 'O'
        head_x, head_y = self.snake[0]
        grid[head_y, head_x] = 'H'
        food_x, food_y = self.food
        grid[food_y, food_x] = 'F'
        print("\n".join([''.join(row) for row in grid]))
        print(f"Score: {len(self.snake) - 1}")