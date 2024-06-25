import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import torch
import sys
from datetime import datetime
import timm
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
import wandb

from dataset import CarDataset, extract_cars_from_bboxes


train_augmentations = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.GaussianBlur(p=0.5),
    A.Blur(p=0.5),
])

val_augmentations = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        # calculated via dataset/dataset.py:calc_norm_constants function
        mean=(0.4179363, 0.39818733, 0.40078917), 
        std=(0.07326048, 0.07070518, 0.07201634),
    ),
    ToTensorV2(),
])

model_name_2_batch_size = {
    "resnet50": 256,
    "efficientnet_es.ra_in1k": 512,
    "tiny_vit_21m_224.dist_in22k_ft_in1k": 256,
    "volo_d1_224.sail_in1k": 224,
}


if __name__ == "__main__":
    data_common_path = "data"

    train_images_path = os.path.join(data_common_path, "train/train")
    test_images_path = os.path.join(data_common_path, "test/test")

    train_bboxes_path = os.path.join(data_common_path, "image_bboxes_train.csv")

    train_filename2image = extract_cars_from_bboxes(train_images_path, train_bboxes_path)

    train_df_path = os.path.join(data_common_path, "train.csv")
    df = pd.read_csv(train_df_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    model_name = "volo_d1_224.sail_in1k"
    model_common_weights_dir_path = "models"
    os.makedirs(model_common_weights_dir_path, exist_ok=True)
    cur_model_weights_dir_path = os.path.join(model_common_weights_dir_path, model_name)
    os.makedirs(cur_model_weights_dir_path, exist_ok=True)

    for i, (train_index, val_index) in enumerate(skf.split(train_df.filename, train_df.label)):
        cur_fold_model_weights_output_path = os.path.join(cur_model_weights_dir_path, f"{model_name}_fold{i}.pth")
        train_train_df = train_df.iloc[train_index]
        train_val_df = train_df.iloc[val_index]

        train_dataset = CarDataset(
            train_train_df,
            train_filename2image,
            train_transforms=train_augmentations,
            val_transforms=val_augmentations,
        )

        val_dataset = CarDataset(
            train_train_df,
            train_filename2image,
            val_transforms=val_augmentations,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=model_name_2_batch_size[model_name], 
            shuffle=True,
            num_workers=12, 
            pin_memory=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=model_name_2_batch_size[model_name], 
            shuffle=False,
            num_workers=10, 
            pin_memory=True,
        )

        model = timm.create_model(model_name, pretrained=True, drop_rate=0.5)
        model.reset_classifier(196)
        model = model.cuda()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=3, 
            min_lr=1e-6,
            verbose=True,
        )
        num_epochs = 30

        with wandb.init(
            config={
                "epochs": num_epochs,
                "model_name": model_name,
            },
            project="intro-dl-2024",
            group=model_name,
            name=str(i),
        ) as run:
            best_val_acc = 0
            for epoch in tqdm.tqdm(range(1, num_epochs + 1)):

                train_accuracy = 0
                train_loss = 0
                total = 0
                correct = 0
                batch_idx = 0

                model.train()

                for inputs, targets in train_loader:
                    inputs, targets = inputs.cuda(), targets.cuda()

                    outputs = model(inputs)

                    loss = ce_loss(outputs, targets)
                    train_loss += loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += predicted.eq(targets.data).cpu().sum()

                    train_accuracy = float(100.0 * correct) / total
                    run.log({
                        "accuracy": train_accuracy,
                        "loss": loss,
                    })

                model.eval()
                total = 0
                correct = 0
                valid_loss = 0
                val_batch = 0
                with torch.no_grad():
                    for img, targets in val_loader:
                        targets = targets.cuda()
                        inputs = img.cuda()

                        outputs = model(inputs)
                        val_loss = ce_loss(outputs, targets)
                        valid_loss += val_loss
                        val_batch += 1

                        _, predicted = torch.max(outputs, 1)

                        total += targets.size(0)
                        correct += predicted.eq(targets.data).cpu().sum()

                    val_accuracy = float(100.0 * correct) / total
                    valid_loss = valid_loss / val_batch

                    lr_scheduler.step(val_accuracy)
                    run.log({
                        "fold_accuracy": val_accuracy,
                        "val_loss": valid_loss,
                    })

                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    torch.save(model.state_dict(), cur_fold_model_weights_output_path)
            
            val_dataset = CarDataset(
                val_df,
                train_filename2image,
                val_transforms=val_augmentations,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=model_name_2_batch_size[model_name], 
                shuffle=False,
                num_workers=10, 
                pin_memory=True,
            )

            best_fold_state_dict = torch.load(cur_fold_model_weights_output_path)
            model.load_state_dict(best_fold_state_dict)
            model.eval()

            total = 0
            correct = 0
            with torch.no_grad():
                for img, targets in val_loader:
                    targets = targets.cuda()
                    inputs = img.cuda()

                    outputs = model(inputs)

                    _, predicted = torch.max(outputs, 1)

                    total += targets.size(0)
                    correct += predicted.eq(targets.data).cpu().sum()

                val_accuracy = float(100.0 * correct) / total

                run.log({
                    "val_accuracy": val_accuracy,
                })
