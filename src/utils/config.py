# Initial configurations
PROJECT_ROOT = '/home/stevenmaio/PycharmProjects/rl-repair'
INSTANCES = 'data/instances/random3sat/small'

# MDP configuration
DISCOUNT_FACTOR = 0.4

# Training configuration
INPUT_MODEL = 'data/torch_models/random3sat-es-serial.pt'
OUTPUT_MODEL = 'data/torch_models/random3sat-es-serial.pt'
NUM_EPOCHS = 1
NUM_TRAJECTORIES = 5
NUM_WORKERS = 5
SAMPLE_INDICES = False
LEARNING_PARAMETER = 0.2
LEARNING_RATE = 0.1
SAVE_RATE = 1
