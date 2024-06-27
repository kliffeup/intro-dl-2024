import albumentations as A
from albumentations.pytorch import ToTensorV2


train_augmentations = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.GaussianBlur(p=0.5),
        A.Blur(p=0.5),
    ]
)

val_augmentations = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(
            # calculated via dataset/dataset.py:calc_norm_constants function
            mean=(0.4179363, 0.39818733, 0.40078917),
            std=(0.07326048, 0.07070518, 0.07201634),
        ),
        ToTensorV2(),
    ]
)

MODEL_NAME_TO_BATCH_SIZE = {
    "resnet50": 256,
    "efficientnet_es.ra_in1k": 512,
    "tiny_vit_21m_224.dist_in22k_ft_in1k": 256,
    "volo_d1_224.sail_in1k": 224,
    "swin_s3_tiny_224.ms_in1k": 224,
    "convformer_s18.sail_in22k_ft_in1k": 256,
    "mobileone_s4.apple_in1k": 256,
    "knn_mobileone_s4.apple_in1k": 256,
}

MODEL_NAME_TO_EMBEDDING_DIM = {
    "mobileone_s4.apple_in1k": 2048,
}

TTA_TRAIN = 2
TTA_INFER = 20
N_NEIGHBORS = 20
