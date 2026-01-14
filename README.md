# Offline reinforcement learning for autonomous driving using the DonkeyCar simulator.

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- DonkeyCar simulator

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd rl-uncontrolled-intersections
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment:**
   ```bash
   export AAE_PATH=results/ae_model.pkl  # Path to trained autoencoder
   ```

### Running the Experiment

**Reproduce main results:**
```bash
python src/run_experiment.py \
    --dataset data/raw/dataset.pkl \
    --ae-path results/ae_model.pkl \
    --n-epochs 10
```

The script will:
1. Load the offline dataset
2. Initialize CQL (Conservative Q-Learning) agent
3. Train for specified epochs
4. Save the trained model to `results/cql_donkey_policy`

### Expected Output

- Trained CQL model saved to `results/cql_donkey_policy`
- Training logs in `results/logs/`
- Model checkpoints during training

## Project Structure

```
├── README.md              # This file
├── requirements.txt       # Exact package versions
├── LICENSE                # MIT License
├── Dockerfile             # Reproducible environment
├── src/
│   ├── algorithms/        # Algorithm implementations
│   │   ├── autoencoder.py      # Autoencoder for state compression
│   │   ├── cql_trainer.py      # CQL offline RL training
│   │   └── config.py           # Configuration
│   ├── environments/      # Environment wrappers
│   │   └── wrapper.py          # Autoencoder wrapper for DonkeyCar
│   └── run_experiment.py  # Main experiment script
├── data/
│   └── raw/              # Dataset files (dataset.pkl)
├── results/
│   ├── logs/             # Training logs
│   └── figures/          # Generated plots
├── notebooks/            # Analysis notebooks
├── tests/                # Unit tests
└── docs/                 # API documentation
```

## Key Components

### Algorithms

- **CQL (Conservative Q-Learning)**: Offline RL algorithm that learns from fixed datasets without environment interaction
- **Autoencoder**: Compresses high-dimensional image observations into low-dimensional latent representations

### Environments

- **DonkeyCar**: Autonomous driving simulator
- **AutoencoderWrapper**: Wraps DonkeyCar environment to use compressed observations

## Training Details

### CQL Hyperparameters

- Batch size: 512
- Conservative weight: 10.0
- Action samples: 20
- Actor learning rate: 3e-4
- Critic learning rate: 3e-4
- Discount factor (γ): 0.99

### Autoencoder

- Latent dimension: 32 (configurable)
- Input size: 96x96x3 RGB images
- Architecture: Convolutional encoder-decoder

## Data Format

The dataset should be an MDPDataset pickle file containing:
- Observations (compressed via autoencoder)
- Actions
- Rewards
- Next observations
- Terminations

## Evaluation

After training, test the model:
```bash
python src/drive_cql.py  # Uses trained CQL model
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Keep code clean and commented

## Troubleshooting

**Issue: AAE_PATH not set**
- Solution: Export the path to your trained autoencoder: `export AAE_PATH=path/to/ae_model.pkl`

**Issue: Dataset not found**
- Solution: Ensure `data/raw/dataset.pkl` exists or provide path with `--dataset`

**Issue: CUDA out of memory**
- Solution: Reduce batch size with `--batch-size 256` or use CPU with `--no-gpu`


## License

MIT License - see LICENSE file for details.
