# API Documentation

## Algorithms

### `src.algorithms.cql_trainer`

#### `train_cql(dataset_path, model_save_path, ...)`

Train a CQL (Conservative Q-Learning) offline RL agent.

**Parameters:**
- `dataset_path` (str): Path to MDPDataset pickle file
- `model_save_path` (str): Path to save trained model
- `batch_size` (int): Batch size for training (default: 512)
- `conservative_weight` (float): Weight for conservative loss term (default: 10.0)
- `n_action_samples` (int): Number of action samples for CQL (default: 20)
- `actor_learning_rate` (float): Learning rate for actor (default: 3e-4)
- `critic_learning_rate` (float): Learning rate for critic (default: 3e-4)
- `gamma` (float): Discount factor (default: 0.99)
- `n_epochs` (int): Number of training epochs (default: 10)
- `use_gpu` (bool): Whether to use GPU (default: True)
- `experiment_name` (str): Name for experiment logging (default: "cql_donkey")

**Returns:**
- Trained CQL model

### `src.algorithms.autoencoder`

#### `Autoencoder(z_size, input_dimension, learning_rate, normalization_mode)`

Autoencoder model for compressing image observations.

**Parameters:**
- `z_size` (int): Latent space dimension
- `input_dimension` (tuple): Input image dimensions (H, W, C)
- `learning_rate` (float): Learning rate for training
- `normalization_mode` (str): Normalization mode ("rl" or "tf")

**Methods:**
- `encode(observation)`: Encode image to latent vector
- `decode(arr)`: Decode latent vector to image
- `save(save_path)`: Save model to file
- `load(load_path)`: Load model from file (classmethod)

#### `load_ae(path, z_size, quantize)`

Load a trained autoencoder from file.

**Parameters:**
- `path` (str): Path to saved autoencoder
- `z_size` (int, optional): Latent dimension (recovered from file if None)
- `quantize` (bool): Whether to quantize model

**Returns:**
- Autoencoder instance

## Environments

### `src.environments.wrapper`

#### `AutoencoderWrapper(env, ae_path)`

Wraps DonkeyCar environment to use autoencoder-compressed observations.

**Parameters:**
- `env` (gym.Env): The gym environment to wrap
- `ae_path` (str, optional): Path to trained autoencoder (or set AAE_PATH env var)

**Methods:**
- `reset()`: Reset environment and return compressed observation
- `step(action)`: Step environment and return compressed observation

**Observation Space:**
- Box shape: `(z_size + 1,)` where z_size is the autoencoder latent dimension
- Includes: compressed image features + speed

## Configuration

### `src.algorithms.config`

Configuration constants for autoencoder and training:

- `INPUT_DIM`: Input image dimensions (96, 96, 3)
- `RAW_IMAGE_SHAPE`: Raw camera image shape
- `ROI`: Region of interest for image cropping
- `CAMERA_HEIGHT`, `CAMERA_WIDTH`: Camera dimensions
