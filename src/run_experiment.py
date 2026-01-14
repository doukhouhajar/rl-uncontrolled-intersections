#!/usr/bin/env python3
"""
This script trains a CQL (Conservative Q-Learning) agent on the DonkeyCar
environment using autoencoder-compressed observations.

Usage:
    python src/run_experiment.py --dataset data/raw/dataset.pkl --ae-path results/ae_model.pkl
"""
import argparse
import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import gym
import gym_donkeycar
from src.algorithms.cql_trainer import train_cql
from src.environments.wrapper import AutoencoderWrapper


def main():
    parser = argparse.ArgumentParser(
        description="Train offline RL agent on DonkeyCar environment"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/raw/dataset.pkl",
        help="Path to MDPDataset pickle file"
    )
    parser.add_argument(
        "--ae-path",
        type=str,
        required=True,
        help="Path to trained autoencoder model (or set AAE_PATH env var)"
    )
    parser.add_argument(
        "--model-save-path",
        type=str,
        default="results/cql_donkey_policy",
        help="Path to save trained CQL model"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for training"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="donkey-mountain-track-v0",
        help="Gym environment name"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=True,
        help="Use GPU for training"
    )
    parser.add_argument(
        "--no-gpu",
        dest="use_gpu",
        action="store_false",
        help="Disable GPU for training"
    )
    
    args = parser.parse_args()
    
    # Set autoencoder path as environment variable for wrapper
    os.environ["AAE_PATH"] = args.ae_path
    
    # Verify dataset exists
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found: {args.dataset}")
        print("Please ensure the dataset file exists or provide correct path with --dataset")
        sys.exit(1)
    
    # Verify autoencoder exists
    if not os.path.exists(args.ae_path):
        print(f"Error: Autoencoder file not found: {args.ae_path}")
        print("Please train an autoencoder first or provide correct path with --ae-path")
        sys.exit(1)
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(args.model_save_path) if os.path.dirname(args.model_save_path) else ".", exist_ok=True)
    
    print("Offline RL Training Experiment")
    print(f"Dataset: {args.dataset}")
    print(f"Autoencoder: {args.ae_path}")
    print(f"Environment: {args.env}")
    print(f"Epochs: {args.n_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"GPU: {args.use_gpu}")
    
    # Train CQL model
    model = train_cql(
        dataset_path=args.dataset,
        model_save_path=args.model_save_path,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        use_gpu=args.use_gpu
    )
    
    print("Training completed successfully!")
    print(f"Model saved to: {args.model_save_path}")
    
    # Optional: Test the trained model
    print("\nTesting trained model...")
    try:
        env = gym.make(args.env)
        env = AutoencoderWrapper(env, ae_path=args.ae_path)
        
        # Rebuild model with environment
        from d3rlpy.algos import CQL
        test_model = CQL(
            batch_size=args.batch_size,
            conservative_weight=10.0,
            n_action_samples=20,
            actor_learning_rate=3e-4,
            critic_learning_rate=3e-4,
            gamma=0.99,
            use_gpu=args.use_gpu
        )
        test_model.build_with_env(env)
        test_model.load_model(args.model_save_path)
        
        obs = env.reset()
        print("Model loaded and tested successfully!")
        print(f"Observation shape: {obs.shape}")
        
    except Exception as e:
        print(f"Warning: Could not test model: {e}")
        print("Model training completed, but testing failed.")


if __name__ == "__main__":
    main()
