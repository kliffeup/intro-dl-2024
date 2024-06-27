import os

import numpy as np
import pandas as pd
import torch
import tqdm
import pickle
from sklearn.neighbors import KNeighborsClassifier

from constants import (
    MODEL_NAME_TO_BATCH_SIZE,
    N_NEIGHBORS,
    TTA_TRAIN,
    train_augmentations,
    val_augmentations,
)
from dataset import CarDataset, extract_cars_from_bboxes
from knn_training.train_feature_extractor import CustomModel


def eval_loader_with_feature_extractor(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    num_augs_per_image: int = TTA_TRAIN,
) -> list[int]:
    X = []
    y = []

    model.eval()
    with torch.no_grad():
        for _ in tqdm.tqdm(range(num_augs_per_image)):
            X.append([])
            y.append([])
            for img, targets in loader:
                inputs = img.cuda()
                targets_ = targets.numpy()

                _, features = model(inputs)

                X[-1].append(features.cpu().numpy())
                y[-1].append(targets_)

            X[-1] = np.concatenate(X[-1], axis=0)
            y[-1] = np.concatenate(y[-1], axis=0)

        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)

    return X, y


if __name__ == "__main__":
    data_common_path = "data"

    train_images_path = os.path.join(data_common_path, "train/train")
    train_bboxes_path = os.path.join(data_common_path, "image_bboxes_train.csv")

    train_filename2image = extract_cars_from_bboxes(
        train_images_path, train_bboxes_path
    )

    train_df_path = os.path.join(data_common_path, "train.csv")

    df = pd.read_csv(train_df_path)

    model_name = "knn_mobileone_s4.apple_in1k"
    model_common_weights_dir_path = "models"
    os.makedirs(model_common_weights_dir_path, exist_ok=True)
    cur_model_weights_dir_path = os.path.join(model_common_weights_dir_path, model_name)
    os.makedirs(cur_model_weights_dir_path, exist_ok=True)

    val_dataset = CarDataset(
        df,
        train_filename2image,
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

    X, y = eval_loader_with_feature_extractor(model, val_loader)

    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights="distance")
    knn.fit(X, y)

    os.makedirs("models/knn", exist_ok=True)
    with open(
        f"models/knn/knn_tta_train_{TTA_TRAIN}_neigh_{N_NEIGHBORS}.pkl", "wb"
    ) as knn_file:
        pickle.dump(knn, knn_file)
