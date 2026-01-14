
from d3rlpy.algos import CQL
from d3rlpy.dataset import MDPDataset

DATA = "dataset.pkl"

dataset = MDPDataset.load(DATA)

model = CQL(
    batch_size=512,
    conservative_weight=10.0,
    n_action_samples=20,
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    gamma=0.99,
    use_gpu=True
)

model.fit(
    dataset.episodes,
    n_epochs=10,
    experiment_name="cql_donkey"
)

model.save_model("cql_donkey_policy")

print("CQL training finished.")
