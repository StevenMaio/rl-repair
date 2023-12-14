import argparse
import csv
import os

from src.utils import LogEvent

EVENT_INDEX = 4
DATA_INDEX = 5

RAW_VAL_KEY = 'raw_val_score'
VAL_KEY = 'val_score'

TXT_EXTENSION = '.txt'
ADAM = 'adam'
GRADIENT_ASCENT = 'gradient-ascent'

ADAM_IDX = 0
GRADIENT_ASCENT_IDX = 1

SMALL_IDX = 0
MEDIUM_IDX = 1
LARGE_IDX = 2

FIELDS = ['strategy', 'small', 'medium', 'large']


def create_name(big_idx, little_idx):
    if big_idx == ADAM_IDX:
        method = ADAM
    else:
        method = GRADIENT_ASCENT
    size = FIELDS[little_idx + 1]
    return f'{method}-{size}'


def create_event_dict(tokens):
    d = {}
    for t in tokens:
        key, value = t.split('=')
        d[key] = float(value)
    return d


def extract_val_data(filepath):
    val_events = []
    with open(filepath) as f:
        data = f.readlines()
    for l in data:
        tokens = l.split()
        if tokens[EVENT_INDEX] == LogEvent.VAL_COMPUTATION:
            val_events.append(create_event_dict(tokens[DATA_INDEX:]))
    return val_events


def process_data(log_dir, output_dir):
    """
    This function goes through the log data and extracts time series data for the
    validation scores, and the final test scores as well.
    """
    problems = os.listdir(log_dir)
    for p in problems:
        directory = os.sep.join([log_dir, p])
        results = [
            [0, 0, 0],
            [0, 0, 0],
        ]
        field_names = [RAW_VAL_KEY, VAL_KEY]

        log_files = list(filter(lambda f: f[-4:] == TXT_EXTENSION,
                                os.listdir(directory)))
        if len(log_files) == 0:
            continue

        results_directory = os.sep.join([output_dir, p])
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        if not os.path.isdir(results_directory):
            os.mkdir(results_directory)

        log_files = list(map(lambda f: (f[:-4], os.sep.join([directory, f])),
                             log_files))
        for title, log in log_files:
            val_events = extract_val_data(log)
            final_test_score: float = 0.00
            big_idx: int
            little_idx: int
            with open(log) as f:
                final_test_score = float(f.readlines()[-1][-5:-1])
            if ADAM in title:
                big_idx = ADAM_IDX
            else:
                big_idx = GRADIENT_ASCENT_IDX
            if 'small' in title:
                little_idx = SMALL_IDX
            elif 'medium' in title:
                little_idx = MEDIUM_IDX
            else:
                little_idx = LARGE_IDX
            results[big_idx][little_idx] = final_test_score
            filename = create_name(big_idx, little_idx)
            with open(os.sep.join([output_dir, p, filename + '.csv']),
                      'w',
                      newline='') as csv_f:
                writer = csv.DictWriter(csv_f, fieldnames=field_names)
                writer.writeheader()

                writer.writerows(val_events)

        # create file with combined test results
        with open(os.sep.join([output_dir, p, 'results.csv']),
                  'w',
                  newline='') as csv_f:
            writer = csv.writer(csv_f)
            adam_data = (ADAM, *results[ADAM_IDX])
            gradient_ascent_data = (GRADIENT_ASCENT, *results[GRADIENT_ASCENT_IDX])
            writer.writerow(FIELDS)
            writer.writerow(adam_data)
            writer.writerow(gradient_ascent_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir')
    parser.add_argument('output_dir')
    args = vars(parser.parse_args())
    process_data(**args)
