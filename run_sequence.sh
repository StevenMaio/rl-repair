export DIR=data/config
export PROBLEMS=covering_one
export TRAINING_METHOD=gradient-ascent10

python main.py train ${DIR}/${PROBLEMS}/${TRAINING_METHOD}-small.json
python main.py train ${DIR}/${PROBLEMS}/${TRAINING_METHOD}-medium.json
python main.py train ${DIR}/${PROBLEMS}/${TRAINING_METHOD}-large.json
