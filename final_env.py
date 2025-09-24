import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from skimage.transform import downscale_local_mean
from einops import repeat

from global_map import local_to_global, GLOBAL_MAP_SHAPE

# Explicitly export public API
__all__ = ["AdvancedPyBoyEnv"]

# Base action lists; we will build the final lists based on config
BASE_PRESS_ACTIONS = [
    WindowEvent.PRESS_ARROW_DOWN,
    WindowEvent.PRESS_ARROW_LEFT,
    WindowEvent.PRESS_ARROW_RIGHT,
    WindowEvent.PRESS_ARROW_UP,
    WindowEvent.PRESS_BUTTON_A,
    WindowEvent.PRESS_BUTTON_B,
    WindowEvent.PRESS_BUTTON_START,
]

BASE_RELEASE_ACTIONS = [
    WindowEvent.RELEASE_ARROW_DOWN,
    WindowEvent.RELEASE_ARROW_LEFT,
    WindowEvent.RELEASE_ARROW_RIGHT,
    WindowEvent.RELEASE_ARROW_UP,
    WindowEvent.RELEASE_BUTTON_A,
    WindowEvent.RELEASE_BUTTON_B,
    WindowEvent.RELEASE_BUTTON_START,
]

event_flags_start = 0xD747
event_flags_end = 0xD87E


class AdvancedPyBoyEnv(gym.Env):
    """Gymnasium-compatible Pokemon Red environment"""
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, config=None, **kwargs):
        super().__init__()

        # Support both a config dict and legacy keyword args (rom_path/state_path/headless)
        self.config = dict(config or {})
        if kwargs:
            # Map legacy names to new ones if present
            legacy_map = {
                "rom_path": "gb_path",
                "state_path": "init_state_path",
            }
            for k, v in kwargs.items():
                mapped = legacy_map.get(k, k)
                self.config[mapped] = v

        self.init_state_path = self.config.get('init_state_path', r'C:\curr_project\open_RL\PPO_trial\has_pokedex.state')
        self.gb_path = self.config.get('gb_path', r'C:\curr_project\open_RL\PPO_trial\PokemonRed.gb')
        self.headless = self.config.get('headless', True)
        self.max_steps = self.config.get('max_steps', 2048)
        self.act_freq = self.config.get('action_freq', 24)
        self.frame_stacks = 3
        self.coords_pad = 12
        self.enc_freqs = 8
        self.render_mode = self.config.get('render_mode', None)
        # Option to allow START/menu; default False to avoid opening settings/menu screens
        self.allow_start = bool(self.config.get('allow_start', False))

        # Initialize PyBoy
        window = "null" if self.headless else "SDL2"
        # Disable audio to prevent buffer overrun errors at high speeds
        self.pyboy = PyBoy(self.gb_path, window=window, sound=False)
        # Emulation speed: 0=unlimited (fast), 1=normal human speed
        # Default to 1 (human-speed) when visible, 0 when headless, but allow override via config
        default_speed = 0 if self.headless else 1
        emu_speed = self.config.get('emulation_speed', default_speed)
        try:
            self.pyboy.set_emulation_speed(int(emu_speed))
            print(f"DEBUG: headless={self.headless}, window={window}, emulation_speed={emu_speed}")
        except Exception as e:
            print(f"DEBUG: Failed to set emulation_speed={emu_speed}: {e}")
            # Fallback silently if API differs
            pass

        # Load event names
        try:
            with open("events.json") as f:
                self.event_names = json.load(f)
        except FileNotFoundError:
            self.event_names = {}

        # Build action lists depending on allow_start
        if self.allow_start:
            self.VALID_ACTIONS = list(BASE_PRESS_ACTIONS)
            self.RELEASE_ACTIONS = list(BASE_RELEASE_ACTIONS)
        else:
            # exclude START (last index)
            self.VALID_ACTIONS = list(BASE_PRESS_ACTIONS[:-1])
            self.RELEASE_ACTIONS = list(BASE_RELEASE_ACTIONS[:-1])

        # Define action space
        self.action_space = spaces.Discrete(len(self.VALID_ACTIONS))
        
        # Enhanced observation space matching RedGymEnv
        self.observation_space = spaces.Dict({
            "screens": spaces.Box(low=0, high=255, shape=(72, 80, self.frame_stacks), dtype=np.uint8),
            "health": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "level": spaces.Box(low=-1, high=1, shape=(self.enc_freqs,), dtype=np.float32),
            "badges": spaces.MultiBinary(8),
            "events": spaces.MultiBinary((event_flags_end - event_flags_start) * 8),
            "map": spaces.Box(low=0, high=255, shape=(self.coords_pad*4, self.coords_pad*4, 1), dtype=np.uint8),
            "recent_actions": spaces.MultiDiscrete([len(self.VALID_ACTIONS)] * self.frame_stacks)
        })
        
        # Initialize environment state
        self.reset_environment_state()
        
        # For random number generation
        self._np_random = None

    def reset_environment_state(self):
        """Initialize all environment tracking variables"""
        self.recent_screens = np.zeros((72, 80, self.frame_stacks), dtype=np.uint8)
        self.recent_actions = np.zeros((self.frame_stacks,), dtype=np.uint8)
        self.step_count = 0
        
        # Exploration tracking
        self.explore_map = np.zeros(GLOBAL_MAP_SHAPE, dtype=np.uint8)
        self.seen_coords = {}
        
        # Event tracking
        self.base_event_flags = 0
        self.current_event_flags_set = {}

    def read_m(self, addr):
        """Read memory at address"""
        return self.pyboy.memory[addr]

    def read_hp_fraction(self):
        """Read current HP fraction across all party members"""
        hp_sum = sum([self.read_hp(addr) for addr in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]])
        max_hp_sum = sum([self.read_hp(addr) for addr in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]])
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        """Read HP value from memory (16-bit)"""
        return 256 * self.read_m(start) + self.read_m(start + 1)
    
    def read_bit(self, addr, bit: int) -> bool:
        """Read specific bit from memory address"""
        return bin(256 + self.read_m(addr))[-bit - 1] == "1"

    def read_event_bits(self):
        """Read all event flag bits as binary array"""
        return [
            int(bit) for i in range(event_flags_start, event_flags_end) 
            for bit in f"{self.read_m(i):08b}"
        ]

    def bit_count(self, bits):
        """Count number of set bits"""
        return bin(bits).count("1")

    def fourier_encode(self, val):
        """Fourier encode continuous values"""
        return np.sin(val * 2 ** np.arange(self.enc_freqs))

    def get_game_coords(self):
        """Get current game coordinates"""
        return (self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E))

    def get_global_coords(self):
        """Convert local coordinates to global map coordinates"""
        x_pos, y_pos, map_n = self.get_game_coords()
        return local_to_global(y_pos, x_pos, map_n)

    def update_seen_coords(self):
        """Track visited coordinates (only when not in battle)"""
        if self.read_m(0xD057) == 0:
            x_pos, y_pos, map_n = self.get_game_coords()
            coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
            if coord_string in self.seen_coords:
                self.seen_coords[coord_string] += 1
            else:
                self.seen_coords[coord_string] = 1

    def update_explore_map(self):
        """Update the global exploration map"""
        c = self.get_global_coords()
        if 0 <= c[0] < self.explore_map.shape[0] and 0 <= c[1] < self.explore_map.shape[1]:
            self.explore_map[c[0], c[1]] = 255

    def get_explore_map(self):
        """Get local view of exploration map around current position"""
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            out = np.zeros((self.coords_pad*2, self.coords_pad*2), dtype=np.uint8)
        else:
            # Handle edge cases where we might go out of bounds
            y_start = max(0, c[0] - self.coords_pad)
            y_end = min(self.explore_map.shape[0], c[0] + self.coords_pad)
            x_start = max(0, c[1] - self.coords_pad)
            x_end = min(self.explore_map.shape[1], c[1] + self.coords_pad)
            
            # Extract the map portion
            map_portion = self.explore_map[y_start:y_end, x_start:x_end]
            
            # Create padded output if needed
            out = np.zeros((self.coords_pad*2, self.coords_pad*2), dtype=np.uint8)
            
            # Calculate where to place the map portion in the output
            out_y_start = self.coords_pad - (c[0] - y_start)
            out_x_start = self.coords_pad - (c[1] - x_start)
            out_y_end = out_y_start + map_portion.shape[0]
            out_x_end = out_x_start + map_portion.shape[1]
            
            # Place the map portion
            out[out_y_start:out_y_end, out_x_start:out_x_end] = map_portion
            
        return repeat(out, 'h w -> (h h2) (w w2)', h2=2, w2=2)

    def update_recent_screens(self, new_screen):
        """Update frame stack with new screen"""
        self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        self.recent_screens[:, :, 0] = new_screen[:, :, 0]

    def update_recent_actions(self, action):
        """Update recent actions buffer"""
        self.recent_actions = np.roll(self.recent_actions, 1)
        self.recent_actions[0] = action

    def get_levels_sum(self):
        """Get normalized sum of party pokemon levels"""
        return 0.02 * sum([
            self.read_m(addr) for addr in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ])

    def get_exploration_reward(self):
        """Simple exploration reward based on unique coordinates visited"""
        return len(self.seen_coords) * 0.01

    def render(self):
        """Render game screen - Gymnasium compatible"""
        if self.render_mode == "rgb_array":
            return self.pyboy.screen.ndarray
        elif self.render_mode == "human":
            # PyBoy handles its own rendering in SDL2 mode
            pass
        # Internal render for observation
        return self._render_for_obs()
    
    def _render_for_obs(self, reduce_res=True):
        """Internal render for observations"""
        game_pixels = self.pyboy.screen.ndarray[:, :, 0:1]
        if reduce_res:
            game_pixels = downscale_local_mean(game_pixels, (2, 2, 1)).astype(np.uint8)
        return game_pixels

    def _get_obs(self):
        """Get current observation"""
        screen = self._render_for_obs()
        self.update_recent_screens(screen)
        
        level_sum = self.get_levels_sum()
        
        observation = {
            "screens": self.recent_screens,
            "health": np.array([self.read_hp_fraction()], dtype=np.float32),
            "level": self.fourier_encode(level_sum).astype(np.float32),
            "badges": np.array([int(bit) for bit in f"{self.read_m(0xD356):08b}"], dtype=np.int8),
            "events": np.array(self.read_event_bits(), dtype=np.int8),
            "map": self.get_explore_map()[:, :, None],
            "recent_actions": self.recent_actions.astype(np.int64)  # Ensure correct dtype
        }
        
        return observation

    def reset(self, seed=None, options=None):
        """Reset environment to initial state - Gymnasium compatible"""
        # Handle random seed
        super().reset(seed=seed)
        
        # Load initial state
        with open(self.init_state_path, "rb") as f:
            self.pyboy.load_state(f)
        
        # Reset all tracking variables
        self.reset_environment_state()
        
        # Set base event flags
        self.base_event_flags = sum([
            self.bit_count(self.read_m(i)) for i in range(event_flags_start, event_flags_end)
        ])
        
        observation = self._get_obs()
        info = {
            'exploration_count': 0,
            'event_flags_set': 0
        }
        
        return observation, info

    def step(self, action):
        """Execute one environment step - Gymnasium compatible"""
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        
        # Execute action
        self._run_action(action)
        self.step_count += 1
        
        # Update tracking
        self.update_recent_actions(action)
        self.update_seen_coords()
        self.update_explore_map()
        
        # Update event flags tracking periodically
        if self.step_count % 100 == 0:
            for address in range(event_flags_start, event_flags_end):
                val = self.read_m(address)
                for idx, bit in enumerate(f"{val:08b}"):
                    if bit == "1":
                        key = f"0x{address:X}-{idx}"
                        if key in self.event_names:
                            self.current_event_flags_set[key] = self.event_names[key]
        
        # Calculate reward
        reward = self.get_exploration_reward()
        
        # Get observation
        observation = self._get_obs()
        
        # Check termination conditions
        terminated = False  # Game over conditions would go here
        truncated = self.step_count >= self.max_steps
        
        # Prepare info dict
        info = {
            'exploration_count': len(self.seen_coords),
            'event_flags_set': len(self.current_event_flags_set),
            'step_count': self.step_count
        }
        
        return observation, reward, terminated, truncated, info

    def _run_action(self, action):
        """Execute action on emulator"""
        # Send button press
        self.pyboy.send_input(self.VALID_ACTIONS[action])
        
        # Hold for specified frames
        for _ in range(self.act_freq):
            self.pyboy.tick()
        
        # Release button
        self.pyboy.send_input(self.RELEASE_ACTIONS[action])

    def close(self):
        """Clean up PyBoy instance"""
        if hasattr(self, 'pyboy'):
            self.pyboy.stop()
            
    def __del__(self):
        """Ensure cleanup on deletion"""
        self.close()