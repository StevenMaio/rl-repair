export DIR=data/config
export PROBLEMS=(
#"k-clique" "random3sat" "partitioning"
"covering_one"
#"covering_two"
"packing_one"
#"packing_two"
)

touch ${LOG_FILE}

# loop over all the problems and run the different problem sets
for p in "${PROBLEMS[@]}"
do

    export LOG_FILE="${p}_results.txt"
    echo ${p}

    echo "${p}: FPR-01"
    python main.py eval ${DIR}/${p}/fpr-01-small.json >> ${LOG_FILE}
    python main.py eval ${DIR}/${p}/fpr-01-medium.json >> ${LOG_FILE}
    python main.py eval ${DIR}/${p}/fpr-01-large.json >> ${LOG_FILE}

    echo "${p}: FPR-02"
    python main.py eval ${DIR}/${p}/fpr-02-small.json >> ${LOG_FILE}
    python main.py eval ${DIR}/${p}/fpr-02-medium.json >> ${LOG_FILE}
    python main.py eval ${DIR}/${p}/fpr-02-large.json >> ${LOG_FILE}

    echo "${p}: FPR-03"
    python main.py eval ${DIR}/${p}/fpr-03-small.json >> ${LOG_FILE}
    python main.py eval ${DIR}/${p}/fpr-03-medium.json >> ${LOG_FILE}
    python main.py eval ${DIR}/${p}/fpr-03-large.json >> ${LOG_FILE}

    echo "${p}: FPR-04"
    python main.py eval ${DIR}/${p}/fpr-04-small.json >> ${LOG_FILE}
    python main.py eval ${DIR}/${p}/fpr-04-medium.json >> ${LOG_FILE}
    python main.py eval ${DIR}/${p}/fpr-04-large.json >> ${LOG_FILE}

    rm ${LOG_FILE}

done
