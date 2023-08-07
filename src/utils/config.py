# Initial configurations
PROJECT_ROOT = '/home/stevenmaio/PycharmProjects/rl-repair'
INSTANCES = 'data/instances/random3sat/small'

# MDP configuration
DISCOUNT_FACTOR = 0.4
MAX_BACKTRACKS = 1

# Training configuration
INPUT_MODEL = 'data/torch_models/random3sat-med-es-serial.pt'
OUTPUT_MODEL = 'data/torch_models/random3sat-med-es-serial-one-instance.pt'
# OUTPUT_MODEL = None
NUM_EPOCHS = 1000
NUM_TRAJECTORIES = 10
NUM_WORKERS = 2
SAMPLE_INDICES = True
LEARNING_PARAMETER = 0.2
LEARNING_RATE = 0.1
BATCH_SIZE = 1

# Adam configuration
FIRST_MOMENT_DECAY_RATE = 0.9
SECOND_MOMENT_DECAY_RATE = 0.99
EPSILON = 1e-8

TRAINING_LOG = "training_log.txt"

# validation trainer configuration
NUM_EVAL_TRAJECTORIES = 10
ITERS_TO_PROGRESS_CHECK = 10
NUM_ALLOWABLE_WORSE_VALS = 5

# data configuration
DATA_SPLIT_SEED = 1133291
VAL_PORTION = 0.2
TEST_PORTION = 0.2

# DES config
INIT_TREND = 0
LEVEL_WEIGHT = 0.2
TREND_WEIGHT = 0.2

# K-Moving Means Config
K_MOVING_MEANS_K = 10

NUM_THREADS = 2