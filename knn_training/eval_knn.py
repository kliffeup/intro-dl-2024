from collections import Counter
import os

import numpy as np
import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier
import torch
import tqdm

from constants import (
    MODEL_NAME_TO_BATCH_SIZE,
    N_NEIGHBORS,
    TTA_TRAIN,
    TTA_INFER,
    train_augmentations,
    val_augmentations,
)
from dataset import CarDataset, extract_cars_from_bboxes
from knn_training.train_feature_extractor import CustomModel


# fix segfault
os.environ["OPENBLAS_NUM_THREADS"] = "64"


def eval_dataset_v2(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    knn: KNeighborsClassifier,
    num_augs_per_image: int = TTA_INFER,
) -> list[int]:
    preds = []
    model.eval()
    with torch.no_grad():
        for _ in tqdm.tqdm(range(num_augs_per_image)):
            preds.append([])
            for img, _ in loader:
                inputs = img.cuda()

                _, features = model(inputs)

                preds[-1].append(features.cpu().numpy())

            preds[-1] = np.concatenate(preds[-1], axis=0)

            buf = []

            for i in tqdm.tqdm(range(0, preds[-1].shape[0], 256)):
                buf.append(knn.predict(preds[-1][i : i + 256])[..., None])

            preds[-1] = np.concatenate(buf, axis=0)

            print(preds[-1].shape)

        preds = np.concatenate(preds, axis=1)

        preds_ = []
        for img_preds in tqdm.tqdm(preds):
            preds_.append(Counter(img_preds).most_common(1)[0][0])

    return preds_


if __name__ == "__main__":
    data_common_path = "data"

    test_images_path = os.path.join(data_common_path, "test/test")
    test_bboxes_path = os.path.join(data_common_path, "image_bboxes_test.csv")

    test_filename2image = extract_cars_from_bboxes(test_images_path, test_bboxes_path)
    test_df_path = os.path.join(data_common_path, "sample_submission.csv")

    df = pd.read_csv(test_df_path)

    model_name = "knn_mobileone_s4.apple_in1k"
    model_common_weights_dir_path = "models"
    os.makedirs(model_common_weights_dir_path, exist_ok=True)
    cur_model_weights_dir_path = os.path.join(model_common_weights_dir_path, model_name)
    os.makedirs(cur_model_weights_dir_path, exist_ok=True)

    val_dataset = CarDataset(
        df,
        test_filename2image,
        train_transforms=train_augmentations,
        val_transforms=val_augmentations,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=MODEL_NAME_TO_BATCH_SIZE[model_name],
        shuffle=False,
        num_workers=10,
        pin_memory=True,
    )

    model = CustomModel("mobileone_s4.apple_in1k")
    model = model.cuda()
    cur_fold_model_weights_output_path = os.path.join(
        cur_model_weights_dir_path, f"{model_name}.pth"
    )
    best_fold_state_dict = torch.load(cur_fold_model_weights_output_path)
    model.load_state_dict(best_fold_state_dict)

    with open(
        f"models/knn/knn_tta_train_{TTA_TRAIN}_neigh_{N_NEIGHBORS}.pkl", "rb"
    ) as knn_file:
        knn = pickle.load(knn_file)

    preds = eval_dataset_v2(model, val_loader, knn)

    submission_path = os.path.join(data_common_path, "sample_submission.csv")
    submission = pd.read_csv(submission_path)
    submission["label"] = preds

    submission_name = (
        f"{model_name}_tta_train_{TTA_TRAIN}_neigh_{N_NEIGHBORS}_tta_infer_{TTA_INFER}"
    )

    submission.to_csv(
        os.path.join(data_common_path, f"{submission_name}_submission.csv"), index=None
    )
