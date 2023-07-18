# Initial configurations
PROJECT_ROOT = '/home/stevenmaio/PycharmProjects/rl-repair'
INSTANCES = 'data/instances/random3sat/small'

# MDP configuration
DISCOUNT_FACTOR = 0.8

# Training configuration
INPUT_MODEL = 'data/torch_models/random3sat-med-es-serial.pt'
OUTPUT_MODEL = 'data/torch_models/random3sat-med-es-serial-one-instance.pt'
# OUTPUT_MODEL = None
NUM_EPOCHS = 100
NUM_TRAJECTORIES = 5
NUM_WORKERS = 5
SAMPLE_INDICES = True
LEARNING_PARAMETER = 0.2
LEARNING_RATE = 0.1
SAVE_RATE = 2
BATCH_SIZE = 12

# Adam configuration
FIRST_MOMENT_DECAY_RATE = 0.9
SECOND_MOMENT_DECAY_RATE = 0.99
EPSILON = 1e-8

TRAINING_LOG = "training_log.txt"

# validation trainer configuration
NUM_VAL_TRAJECTORIES = 5
ITERS_TO_VAL = 5
NUM_ALLOWABLE_WORSE_VALS = 10

# data configuration
DATA_SPLIT_SEED = 1133291
VAL_PORTION = 0.2
TEST_PORTION = 0.2
