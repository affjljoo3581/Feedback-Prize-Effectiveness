from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import GroupKFold

LABEL_NAMES = ["Ineffective", "Adequate", "Effective"]


def main(args: argparse.Namespace):
    with open(args.params) as fp:
        params = json.load(fp)

    oof_dataset = pd.read_csv(args.oof, index_col="discourse_id")
    oof_dataset.discourse_type = oof_dataset.discourse_type.astype("category")
    oof_dataset.discourse_effectiveness = oof_dataset.discourse_effectiveness.apply(
        LABEL_NAMES.index
    )

    test_dataset = pd.read_csv(args.test, index_col="discourse_id")
    test_dataset.discourse_type = test_dataset.discourse_type.astype("category")
    test_dataset = test_dataset.drop("essay_id", axis=1)

    probabilities, best_scores = [], []
    splits = GroupKFold(10).split(oof_dataset, groups=oof_dataset.essay_id)
    for train_indices, val_indices in splits:
        train_oof = oof_dataset.iloc[train_indices]
        X_train = train_oof.drop(["essay_id", "discourse_effectiveness"], axis=1)
        y_train = train_oof.discourse_effectiveness

        val_oof = oof_dataset.iloc[val_indices]
        X_val = val_oof.drop(["essay_id", "discourse_effectiveness"], axis=1)
        y_val = val_oof.discourse_effectiveness

        classifier = LGBMClassifier(n_estimators=1000, **params)
        classifier.fit(
            X_train,
            y_train,
            eval_metric="logloss",
            eval_set=(X_val, y_val),
            callbacks=[early_stopping(100)],
        )
        probabilities.append(classifier.predict_proba(test_dataset))
        best_scores.append(classifier.best_score_["valid_0"]["multi_logloss"])

    print(f"[*] cv score: {np.mean(best_scores)}")

    probabilities = np.stack(probabilities).mean(0)
    probabilities = {name: probabilities[:, i] for i, name in enumerate(LABEL_NAMES)}
    submission = pd.DataFrame({"discourse_id": test_dataset.index, **probabilities})
    submission.to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("oof")
    parser.add_argument("test")
    parser.add_argument("--params", default="params.json")
    parser.add_argument("--output", default="submission.csv")
    main(parser.parse_args())
