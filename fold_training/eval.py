from collections import Counter
import os

import numpy as np
import pandas as pd
import torch
import timm
import tqdm

from constants import MODEL_NAME_TO_BATCH_SIZE, train_augmentations, val_augmentations
from dataset import CarDataset, extract_cars_from_bboxes


def eval_dataset(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    num_augs_per_image: int = 21,
) -> list[int]:
    preds = []
    model.eval()
    with torch.no_grad():
        for _ in tqdm.tqdm(range(num_augs_per_image)):
            preds.append([])
            for img, _ in loader:
                inputs = img.cuda()

                outputs = model(inputs)

                _, predicted = torch.max(outputs, 1)
                preds[-1].append(predicted.cpu().numpy())

            preds[-1] = np.concatenate(preds[-1], axis=None)[..., None]

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

    model_name = "mobileone_s4.apple_in1k"
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

    total_preds = []
    for i in range(4):
        model = timm.create_model(model_name, pretrained=True, drop_rate=0.5)
        model.reset_classifier(196)
        model = model.cuda()
        cur_fold_model_weights_output_path = os.path.join(
            cur_model_weights_dir_path, f"{model_name}_fold{i}.pth"
        )
        best_fold_state_dict = torch.load(cur_fold_model_weights_output_path)
        model.load_state_dict(best_fold_state_dict)

        fold_predicitons = eval_dataset(model, val_loader)
        total_preds.append(fold_predicitons)

    gt_preds = [target for _, target in val_dataset]

    output_df = pd.DataFrame.from_dict(
        {str(i): fold_predicitons for i, fold_predicitons in enumerate(total_preds)}
        | {"label": gt_preds}
    )

    output_df.to_csv(os.path.join(data_common_path, f"{model_name}_test.csv"))
