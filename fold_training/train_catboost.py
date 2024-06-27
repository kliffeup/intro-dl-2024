import os

import catboost as cb
import pandas as pd


if __name__ == "__main__":
    data_common_path = "data"
    model_name = "mobileone_s4.apple_in1k"

    train_df = pd.read_csv(os.path.join(data_common_path, f"{model_name}_val.csv"))

    features = train_df.iloc[:, :-1].values
    label = train_df.iloc[:, -1].values

    train_data = cb.Pool(
        data=features,
        label=label,
        cat_features=[0, 1, 2, 3],
    )

    model = cb.CatBoostClassifier(
        iterations=300,
        random_state=42,
        classes_count=196,
        eval_metric="Accuracy",
        task_type="GPU",
        devices="0",
    )
    model.fit(train_data)

    test_df = pd.read_csv(os.path.join(data_common_path, f"{model_name}_test.csv"))

    test_data = cb.Pool(
        data=test_df.iloc[:, :-1].values,
        cat_features=[0, 1, 2, 3],
    )

    submission_path = os.path.join(data_common_path, "sample_submission.csv")
    submission = pd.read_csv(submission_path)
    submission["label"] = model.predict(test_data)

    submission.to_csv(
        os.path.join(data_common_path, f"{model_name}_catboost_submission.csv"),
        index=None,
    )
