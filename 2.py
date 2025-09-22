import numpy as np
import random
import time
from pyboy import PyBoy

class PokemonEnv:
    def __init__(self, rom_path, state_path, headless=True):
        window = "null" if headless else "SDL2"
        self.pyboy = PyBoy(rom_path, window=window, sound=False)
        self.wrapper = self.pyboy.game_wrapper
        self.state_path = state_path

        self.actions = ['down', 'up', 'left', 'right']
        self.state = None
        self.reset()

    def _extract_state(self):
        area = self.wrapper.game_area()
        small = area[::8, ::8]
        return tuple((small // 64).flatten())

    def reset(self):
        with open(self.state_path, "rb") as f:
            self.pyboy.load_state(f)
        self.state = self._extract_state()
        return self.state, {}

    def step(self, action):
        for _ in range(20):
            self.pyboy.button(self.actions[action])
            self.pyboy.tick()

        next_state = self._extract_state()
        reward = 1 if next_state != self.state else 0
        self.state = next_state
        done = self.wrapper.game_over
        return next_state, reward, done, {}

    def close(self):
        self.pyboy.stop()


def q_learning(env, episodes=50, alpha=0.1, gamma=0.99, epsilon=0.5, max_no_reward_steps=50):
    q_table = {}
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        no_reward_counter = 0

        while True:
            if state not in q_table:
                q_table[state] = np.zeros(len(env.actions))

            if random.random() < epsilon:
                action = random.randint(0, len(env.actions) - 1)
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, _ = env.step(action)

            if next_state not in q_table:
                q_table[next_state] = np.zeros(len(env.actions))

            q_table[state][action] = (1 - alpha) * q_table[state][action] + \
                                     alpha * (reward + gamma * np.max(q_table[next_state]))

            state = next_state
            total_reward += reward

            if reward == 0:
                no_reward_counter += 1
            else:
                no_reward_counter = 0

            if no_reward_counter >= max_no_reward_steps or done:
                break

        print(f"Episode {ep+1}/{episodes} - Total Reward: {total_reward}")

    return q_table


def run_agent(env, q_table, delay=0.05, max_no_reward_steps=50):
    state, _ = env.reset()
    no_reward_counter = 0

    while True:
        if state in q_table:
            action = int(np.argmax(q_table[state]))
        else:
            action = random.randint(0, len(env.actions) - 1)

        for _ in range(20):
            env.pyboy.button(env.actions[action])
            env.pyboy.tick()

        next_state = env._extract_state()
        reward = 1 if next_state != state else 0

        if reward == 0:
            no_reward_counter += 1
        else:
            no_reward_counter = 0

        state = next_state
        time.sleep(delay)

        if no_reward_counter >= max_no_reward_steps or env.wrapper.game_over:
            print("Episode ended due to inactivity or game over!")
            break

    env.close()


if __name__ == "__main__":
    env = PokemonEnv("PokemonRed.gb", "pallet.state", headless=True)
    q_table = q_learning(env, episodes=50, max_no_reward_steps=500)  # fewer episodes for testing
    env.close()

    env = PokemonEnv("PokemonRed.gb", "pallet.state", headless=False)
    run_agent(env, q_table, delay=0.2, max_no_reward_steps=500)

