import sys
from os.path import exists
from pathlib import Path

# Add parent directory to path if needed
sys.path.append('..')

# Try importing with error handling
try:
    from final_env import AdvancedPyBoyEnv
except ImportError:
    print("Could not import AdvancedPyBoyEnv from final_env")
    print("Checking if file exists...")
    if exists("final_env.py"):
        print("final_env.py exists. Trying direct import...")
        # If the class name is different, we'll need to check
        import final_env
        print("Available classes/functions in final_env:", dir(final_env))
    else:
        print("final_env.py not found in current directory")
    sys.exit(1)

from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
import torch

def make_env(rank, env_config, seed=0):
    """
    Utility function for multiprocessed env
    """
    def _init():
        # Create environment with config
        env = AdvancedPyBoyEnv(env_config)
        # Set seed if environment supports it
        set_random_seed(seed + rank)
        return env
    
    return _init

if __name__ == "__main__":
    # Configuration
    use_wandb_logging = False
    HEADLESS = False
    # Shorter episodes when watching so it's easier to observe
    ep_length = 2048 * 2 if not HEADLESS else 2048 * 20
    sess_id = "enhanced_ppo_runs"
    sess_path = Path(sess_id)
    sess_path.mkdir(exist_ok=True)
    
    # Environment configuration
    env_config = {
        'headless': HEADLESS,
        # Much faster when watching; 0 = unlimited in headless mode
        'emulation_speed': (0 if HEADLESS else 5),
        'allow_start': False,  # disallow START/menu
        'action_freq': 8,  # Faster action frequency for more responsive speed
        'init_state_path': r'has_pokedex.state',
        'gb_path': r'PokemonRed.gb',
        'max_steps': ep_length,
    }
    
    print("=" * 50)
    print("Pokemon Red PPO Training")
    print("=" * 50)
    print(f"Environment Config: {env_config}")
    print(f"Episode Length: {ep_length}")
    print(f"Session Path: {sess_path}")
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Create vectorized environment
    num_cpu = 1 if not HEADLESS else 9
    print(f"\nCreating {num_cpu} parallel environments...")
    if not HEADLESS:
        # Always use single-process env for a visible window on Windows
        env = DummyVecEnv([make_env(0, env_config)])
        print("Using DummyVecEnv (single process, visible window)")
    else:
        try:
            # Try SubprocVecEnv first (faster but can have issues)
            env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
            print("Using SubprocVecEnv (multiprocess)")
        except Exception as e:
            print(f"SubprocVecEnv failed: {e}")
            print("Falling back to DummyVecEnv (single process)")
            envs = [AdvancedPyBoyEnv(env_config) for _ in range(num_cpu)]
            env = DummyVecEnv([lambda: e for e in envs])
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=ep_length // 2,  # Save every half episode
        save_path=str(sess_path),
        name_prefix="enhanced_poke"
    )
    
    callbacks = [checkpoint_callback]
    
    # Calculate batch size
    train_steps_batch = ep_length // num_cpu
    batch_size = min(128, train_steps_batch // 4)  # Ensure batch size is reasonable
    
    model_config = {
        "policy": "MultiInputPolicy",
        "env": env,
        "verbose": 1,
        "n_steps": train_steps_batch,
        "batch_size": batch_size,
        "n_epochs": 3,
        "gamma": 0.998,  # High gamma for long-term rewards
        "gae_lambda": 0.95,
        "ent_coef": 0.01,  # Exploration bonus
        "learning_rate": 2.5e-4,
        "clip_range": 0.2,
        "tensorboard_log": str(sess_path),
        "device": device
    }
    
    print(f"Model Configuration:")
    for key, value in model_config.items():
        if key not in ["env", "policy"]:
            print(f"  {key}: {value}")
    
    model = PPO(**model_config)
    
    print(f"\nModel Policy: {model.policy}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    
    # Training
    print("\n=== Starting Training ===")
    total_timesteps = ep_length * num_cpu * (5 if not HEADLESS else 50)  # shorter when watching
    print(f"Total training timesteps: {total_timesteps:,}")
    print(f"Estimated episodes: {total_timesteps // ep_length}")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            tb_log_name="enhanced_poke_ppo",
            progress_bar=True  # Show progress bar
        )
        
        # Save final model
        final_model_path = sess_path / "final_model"
        model.save(final_model_path)
        print(f"\n✅ Training completed! Model saved to {final_model_path}")
        
        # Save training stats
        print("\n=== Training Summary ===")
        print(f"Total timesteps trained: {total_timesteps:,}")
        print(f"Model saved to: {final_model_path}")
        print(f"Checkpoints saved in: {sess_path}")
        print(f"Tensorboard logs in: {sess_path}")
        print("\nTo view tensorboard logs, run:")
        print(f"  tensorboard --logdir {sess_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        save_path = sess_path / "interrupted_model"
        model.save(save_path)
        print(f"Model saved to {save_path}")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        save_path = sess_path / "error_model"
        try:
            model.save(save_path)
            print(f"Partial model saved to {save_path}")
        except:
            print("Could not save model")
    finally:
        # Cleanup
        env.close()
        print("\nEnvironments closed")
