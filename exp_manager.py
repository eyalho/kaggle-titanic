import csv
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from pipes_transformers.baseline import BaselineV1Transformer, BaselineV2Transformer, BaselineV3Transformer, \
    BaselineV4Transformer


def load_data(base_dir=Path(".")):
    df_train = pd.read_csv(base_dir / "train.csv")
    df_test = pd.read_csv(base_dir / "test.csv")
    df_anno_example = pd.read_csv(base_dir / "gender_submission.csv")
    return df_train, df_test, df_anno_example


def split_to_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test


def evaluate_model(X, pipe, y_true):
    LOG_FILE_PATH = "experiments_results.csv"
    y_predicted = pipe.predict(X)
    print("f1_score macro", f1_score(y_true, y_predicted, average='binary'))
    print("accuracy_score", accuracy_score(y_true, y_predicted))
    ordered_fieldnames = OrderedDict([('pipe_name', None), ('f1_score', None), ('accuracy_score', None)])
    with open(LOG_FILE_PATH, "a") as f:
        dw = csv.DictWriter(f, fieldnames=ordered_fieldnames)
        if f.tell() == 0:
            dw.writeheader()
        dw.writerow({
            'pipe_name': get_pipe_name(pipe),
            'f1_score': f1_score(y_true, y_predicted, average='binary'),
            'accuracy_score': accuracy_score(y_true, y_predicted)
        })


def create_submission_file(X, pipe):
    SUB_DIR = Path("submission_files")
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    y_submission = pipe.predict(X)
    df = pd.DataFrame({"PassengerId": X['PassengerId'], "Survived": y_submission}, columns=['PassengerId', 'Survived'])
    df.to_csv(SUB_DIR / f"{get_pipe_name(pipe)}.csv", index=False)


def get_pipe_name(pipe):
    return "--".join(pipe.steps[i][0] for i in range(len(pipe.steps)))


def run_full_experiment(pipe, repeat=3):
    for i in range(repeat):
        print("-"*20)
        print(f"run experiment {get_pipe_name(pipe)}")

        df_train, df_test, df_anno_example = load_data()
        X_train, X_test, y_train, y_test = train_test_split(df_train.copy(), df_train.copy().Survived)

        pipe.fit(X_train, y_train)
        evaluate_model(X_test, pipe, y_test)

        create_submission_file(df_test, pipe)


# kaggle competitions submit -c titanic -f %d_baseline.csv -m "%d_baseline"
if __name__ == "__main__":
    run_full_experiment(Pipeline([('baseline_v1', BaselineV1Transformer()), ('RF', RandomForestClassifier())]))
    run_full_experiment(Pipeline([('baseline_v2', BaselineV2Transformer()), ('RF', RandomForestClassifier())]))
    run_full_experiment(Pipeline([('baseline_v3', BaselineV3Transformer()), ('RF', RandomForestClassifier())]))
    run_full_experiment(Pipeline([('baseline_v4', BaselineV4Transformer()), ('RF', RandomForestClassifier())]))


