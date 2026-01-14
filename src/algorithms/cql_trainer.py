from d3rlpy.algos import CQL
from d3rlpy.dataset import MDPDataset


def train_cql(
    dataset_path: str,
    model_save_path: str = "cql_donkey_policy",
    batch_size: int = 512,
    conservative_weight: float = 10.0,
    n_action_samples: int = 20,
    actor_learning_rate: float = 3e-4,
    critic_learning_rate: float = 3e-4,
    gamma: float = 0.99,
    n_epochs: int = 10,
    use_gpu: bool = True,
    experiment_name: str = "cql_donkey"
):
    
    print(f"Loading dataset from {dataset_path}...")
    dataset = MDPDataset.load(dataset_path)

    print("Initializing CQL model...")
    model = CQL(
        batch_size=batch_size,
        conservative_weight=conservative_weight,
        n_action_samples=n_action_samples,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        gamma=gamma,
        use_gpu=use_gpu
    )

    print(f"Training for {n_epochs} epochs...")
    model.fit(
        dataset.episodes,
        n_epochs=n_epochs,
        experiment_name=experiment_name
    )

    print(f"Saving model to {model_save_path}...")
    model.save_model(model_save_path)

    print("CQL training finished.")
    return model
