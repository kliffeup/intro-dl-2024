import os
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, BatchSampler
from sklearn.model_selection import train_test_split
import wandb
import tqdm


from constants import (
    MODEL_NAME_TO_BATCH_SIZE,
    MODEL_NAME_TO_EMBEDDING_DIM,
    train_augmentations,
    val_augmentations,
)
from dataset import CarDataset, extract_cars_from_bboxes


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 2) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        distances = (outputs.unsqueeze(1) - outputs.unsqueeze(0)).pow(2).sum(2)
        label_equal = targets.unsqueeze(1) == targets.unsqueeze(0)

        positive_distances = distances * label_equal
        negative_distances = distances * ~label_equal

        positive_loss = positive_distances.mean()
        negative_loss = F.relu(self.margin - negative_distances).mean()

        return positive_loss + negative_loss


class CustomBatchSampler(Sampler):
    def __init__(
        self,
        data_source: CarDataset,
        elems_per_class: int = 8,
        classes_per_batch: int = 32,
    ):
        self.data_source = data_source
        self.elems_per_class = elems_per_class
        self.classes_per_batch = classes_per_batch

    def __iter__(self):
        for _ in range(len(self)):
            selected_classes = np.random.choice(
                len(self.data_source.classes), self.classes_per_batch, replace=False
            )
            batch = []
            for class_label in selected_classes:
                batch.extend(
                    np.random.choice(
                        self.data_source.classes_to_samples[class_label],
                        self.elems_per_class,
                        replace=len(self.data_source.classes_to_samples[class_label])
                        < self.elems_per_class,
                    )
                )

            np.random.shuffle(batch)
            for idx in batch:
                yield idx

    def __len__(self):
        return len(self.data_source) // (self.classes_per_batch * self.elems_per_class)


class CustomModel(nn.Module):
    def __init__(
        self, model_name: str, num_classes: int = 196, p_dropout: float = 0.2
    ) -> None:
        super().__init__()

        self.feature_extractor = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
        )

        self.head = nn.Sequential(
            nn.Dropout(p=p_dropout, inplace=False),
            nn.Linear(MODEL_NAME_TO_EMBEDDING_DIM[model_name], num_classes),
        )

    def forward(self, input_tensor: torch.Tensor):
        features = self.feature_extractor(input_tensor)
        output = self.head(features)

        return output, features


if __name__ == "__main__":
    data_common_path = "data"

    train_images_path = os.path.join(data_common_path, "train/train")
    train_bboxes_path = os.path.join(data_common_path, "image_bboxes_train.csv")

    train_filename2image = extract_cars_from_bboxes(
        train_images_path, train_bboxes_path
    )

    train_df_path = os.path.join(data_common_path, "train.csv")
    df = pd.read_csv(train_df_path)

    train_df, val_df = train_test_split(
        df, test_size=0.1, random_state=42, stratify=df["label"]
    )

    train_df = train_df.reset_index()
    val_df = train_df.reset_index()

    model_name = "knn_mobileone_s4.apple_in1k"
    model_common_weights_dir_path = "models"
    os.makedirs(model_common_weights_dir_path, exist_ok=True)
    cur_model_weights_dir_path = os.path.join(model_common_weights_dir_path, model_name)
    os.makedirs(cur_model_weights_dir_path, exist_ok=True)

    cur_fold_model_weights_output_path = os.path.join(
        cur_model_weights_dir_path, f"{model_name}.pth"
    )

    train_dataset = CarDataset(
        train_df,
        train_filename2image,
        train_transforms=train_augmentations,
        val_transforms=val_augmentations,
        is_train=True,
    )

    batch_sampler = BatchSampler(
        CustomBatchSampler(
            train_dataset,
            elems_per_class=8,
            classes_per_batch=MODEL_NAME_TO_BATCH_SIZE[model_name] // 8,
        ),
        batch_size=MODEL_NAME_TO_BATCH_SIZE[model_name],
        drop_last=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=12,
        pin_memory=True,
    )

    val_dataset = CarDataset(
        val_df,
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

    model = CustomModel("mobileone_s4.apple_in1k", num_classes=df.label.max() + 1)
    model = model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    contrastive_loss = ContrastiveLoss()

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
        name=model_name,
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

                outputs, features = model(inputs)

                cur_ce_loss = ce_loss(outputs, targets)
                cur_contrastive_loss = contrastive_loss(features, targets)

                loss = cur_ce_loss + cur_contrastive_loss

                train_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                train_accuracy = float(100.0 * correct) / total
                run.log(
                    {
                        "accuracy": train_accuracy,
                        "loss": loss,
                        "ce_loss": cur_ce_loss,
                        "contrastive_loss": cur_contrastive_loss,
                    }
                )

            model.eval()
            total = 0
            correct = 0
            valid_loss = 0
            valid_ce_loss = 0
            valid_contrastive_loss = 0
            val_batch = 0
            with torch.no_grad():
                for img, targets in val_loader:
                    targets = targets.cuda()
                    inputs = img.cuda()

                    outputs, features = model(inputs)

                    cur_ce_loss = ce_loss(outputs, targets)
                    cur_contrastive_loss = contrastive_loss(features, targets)

                    val_loss = cur_ce_loss + cur_contrastive_loss

                    valid_loss += val_loss
                    valid_ce_loss += cur_ce_loss
                    valid_contrastive_loss += cur_contrastive_loss
                    val_batch += 1

                    _, predicted = torch.max(outputs, 1)

                    total += targets.size(0)
                    correct += predicted.eq(targets.data).cpu().sum()

                val_accuracy = float(100.0 * correct) / total
                valid_loss /= val_batch
                valid_ce_loss /= val_batch
                valid_contrastive_loss /= val_batch

                lr_scheduler.step(val_accuracy)
                run.log(
                    {
                        "val_accuracy": val_accuracy,
                        "val_loss": valid_loss,
                        "val_ce_loss": valid_ce_loss,
                        "val_contrastive_loss": valid_contrastive_loss,
                    }
                )

            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                torch.save(model.state_dict(), cur_fold_model_weights_output_path)
