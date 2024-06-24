import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import sys

class SnakeEnv(gym.Env):
    def __init__(self, grid_size=10, cell_size=40):
        super(SnakeEnv, self).__init__()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.action_space = spaces.Discrete(4)  # 0: up, 1: right, 2: down, 3: left
        self.observation_space = spaces.Box(low=0, high=2, shape=(grid_size, grid_size), dtype=np.uint8)
        
        self.snake = [(grid_size // 2, grid_size // 2)]
        self.food = None
        self.direction = 1
        self.steps = 0
        self.max_steps = 100 * grid_size

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((grid_size * cell_size, grid_size * cell_size))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()

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
        new_head = (head_x + dx, head_y + dy)
        
        # Check if the new head is outside the grid
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            return  # The game will end due to collision in the next step
        
        if new_head == self.food:
            self.snake = [new_head] + self.snake
        else:
            self.snake = [new_head] + self.snake[:-1]

    def _is_collision(self):
        head_x, head_y = self.snake[0]
        # Check collision with walls
        if (head_x < 0 or head_x >= self.grid_size or
            head_y < 0 or head_y >= self.grid_size):
            return True
        # Check collision with self
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
        self.screen.fill((0, 0, 0))  # Fill screen with black

        # Draw snake
        for segment in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0), (segment[0] * self.cell_size, segment[1] * self.cell_size, self.cell_size, self.cell_size))

        # Draw food
        pygame.draw.rect(self.screen, (255, 0, 0), (self.food[0] * self.cell_size, self.food[1] * self.cell_size, self.cell_size, self.cell_size))

        pygame.display.flip()
        self.clock.tick(10)  # Control game speed

    def close(self):
        pygame.quit()

    def manual_play(self):
        done = False
        total_reward = 0
        action = 1  # Start moving right

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return total_reward
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP and action != 2:
                        action = 0
                    elif event.key == pygame.K_RIGHT and action != 3:
                        action = 1
                    elif event.key == pygame.K_DOWN and action != 0:
                        action = 2
                    elif event.key == pygame.K_LEFT and action != 1:
                        action = 3

            _, reward, done, _, _ = self.step(action)
            total_reward += reward
            self.render()
            print(f"Score: {len(self.snake) - 1}", end='\r')

        print(f"\nGame Over! Final Score: {len(self.snake) - 1}")
        return total_reward