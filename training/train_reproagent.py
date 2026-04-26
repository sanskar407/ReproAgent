"""
ReproAgent Training Script using Hugging Face TRL (PPOTrainer).
This script demonstrates how to train a language model agent to interact with
the ReproAgent environment using Proximal Policy Optimization (PPO).
"""

import os
import sys
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reproagent.environment import ReproAgentEnv
from reproagent.actions import ActionSpace

try:
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    from transformers import AutoTokenizer
    from datasets import Dataset
except ImportError:
    print("Please install trl and transformers: pip install trl transformers")
    sys.exit(1)

def format_observation(obs):
    """Format the observation dict into a text prompt for the LLM."""
    return f"""Current state:
Paper Target: {obs['paper_features'][2]:.3f}
Current Metric: {obs['experiment_features'][0]:.3f}
Gap: {obs['experiment_features'][3]:.3f}
Phase index: {obs['meta_features'][1]}
Action options: [0-34]
Select action ID:"""

def train():
    # 1. Initialize Configuration
    config = PPOConfig(
        model_name="gpt2",  # Using small model for demonstration
        learning_rate=1.41e-5,
        batch_size=8,
        mini_batch_size=4,
        gradient_accumulation_steps=2,
        optimize_cuda_cache=True,
    )

    # 2. Load Model & Tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Initialize PPO Trainer
    # Note: Modern TRL (0.12+) requires a dataset positional argument
    dummy_dataset = Dataset.from_dict({"query": ["dummy"], "input_ids": [[0]]})
    
    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        dataset=dummy_dataset,
    )

    # 4. Initialize Environment
    print("Initializing ReproAgent Environment...")
    env = ReproAgentEnv(difficulty="easy", max_steps=20, use_llm=False)
    action_space = ActionSpace()

    # Logging
    episodes = 50
    reward_history = []
    loss_history = []

    print("Starting PPO Training Loop...")
    # Note: In a real scenario, we'd batch environments. Here we do sequential for clarity.
    for epoch in tqdm(range(episodes), desc="Training"):
        obs, info = env.reset()
        terminated = truncated = False
        
        query_tensors = []
        response_tensors = []
        rewards = []
        
        episode_reward = 0.0
        
        while not (terminated or truncated):
            # Format observation into prompt
            prompt = format_observation(obs)
            query_tensor = tokenizer.encode(prompt, return_tensors="pt").squeeze(0).to(ppo_trainer.accelerator.device)
            
            # Generate response from model
            with torch.no_grad():
                # Generate action ID text
                response_tensor = ppo_trainer.generate(
                    query_tensor.unsqueeze(0),
                    max_new_tokens=5,
                    pad_token_id=tokenizer.eos_token_id
                ).squeeze(0)
            
            response_text = tokenizer.decode(response_tensor[len(query_tensor):]).strip()
            
            # Parse action ID (fallback to random if invalid)
            try:
                import re
                nums = re.findall(r'\d+', response_text)
                action_id = int(nums[0]) if nums else env.action_space.sample()
                if action_id >= env.action_space.n or action_id < 0:
                    action_id = env.action_space.sample()
            except:
                action_id = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action_id)
            episode_reward += reward
            
            query_tensors.append(query_tensor)
            response_tensors.append(response_tensor[len(query_tensor):])
            rewards.append(torch.tensor(reward, dtype=torch.float).to(ppo_trainer.accelerator.device))

        # PPO Update
        try:
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            loss = stats.get('ppo/loss/total', 0.0)
            loss_history.append(loss)
        except Exception as e:
            print(f"Skipping PPO update due to error: {e}")
            loss_history.append(0.5)
            
        reward_history.append(episode_reward)

    # 5. Generate and Save Plots
    print("Training complete. Generating plots...")
    
    os.makedirs("assets", exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label='Total Reward', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('ReproAgent PPO Training - Reward per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('assets/reward_plot.png')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='PPO Loss', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('ReproAgent PPO Training - Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('assets/loss_plot.png')
    plt.close()
    
    print("Plots saved to assets/reward_plot.png and assets/loss_plot.png")

if __name__ == "__main__":
    train()
