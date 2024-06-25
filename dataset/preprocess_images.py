import multiprocessing as mp
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import DetaImageProcessor, DetaForObjectDetection

parser = ArgumentParser('Get car bounding boxes for images in dataset')
parser.add_argument(
    '--images_path', type=str, help='Path to directory with images',
)
parser.add_argument(
    '--output', '-o', type=str, default='image_boxes.csv', help='Path to output csv with boxes',
)
parser.add_argument(
    '--batch_size', '-bs', type=int, default=8, help='Batch size for batched image processing',
)
parser.add_argument(
    '--device', '-d', type=str, default='cpu', help='Device to process images on',
)


def load_image(images_root: str, filename: str):
    image_path = os.path.join(images_root, filename)
    image = Image.open(image_path)

    np_image = np.array(image)
    if len(np_image.shape) == 2:  # grayscale case
        np_image = np.repeat(np_image[..., None], 3, -1)
        image = Image.fromarray(np_image)

    return image


def df_chunker(df: pd.DataFrame, chunk_size: int):
    for pos in range(0, len(df), chunk_size):
        yield df.iloc[pos:pos + chunk_size]


def process_result(result):
    max_area = 0.0
    max_area_box = None

    for box_score_threshold in [0.3, 0.1, 0.01]:
        for score, label, (xmin, ymin, xmax, ymax) in zip(
                result['scores'].tolist(), result['labels'].tolist(), result['boxes'].tolist(),
        ):
            if score < box_score_threshold:
                continue
            if label not in [3, 6, 8]:  # ['car', 'truck', 'bus'] in COCO labels
                continue
            area = (xmax - xmin) * (ymax - ymin)
            if area > max_area:
                max_area = area
                max_area_box = (xmin, ymin, xmax, ymax)

        if max_area_box is not None:
            break

    return max_area_box


if __name__ == '__main__':
    args = parser.parse_args()

    images_root = args.images_path
    batch_size = args.batch_size
    device = torch.device(args.device)

    images_df = pd.DataFrame({'filename': os.listdir(images_root)})

    pool = mp.Pool(batch_size)

    image_processor = DetaImageProcessor.from_pretrained('jozhang97/deta-resnet-50-24-epochs')
    object_detection_model = DetaForObjectDetection.from_pretrained('jozhang97/deta-resnet-50-24-epochs').to(device)
    object_detection_model.eval()

    boxes = {}
    for rows in tqdm(df_chunker(images_df, batch_size), total=(len(images_df) - 1) // batch_size + 1):
        images = list(pool.starmap(load_image, [(images_root, filename) for filename in rows['filename']]))

        encoding = image_processor(images, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = object_detection_model(**encoding)
        target_sizes = torch.tensor([image.size[::-1] for image in images])

        results = image_processor.post_process_object_detection(outputs, threshold=0.01, target_sizes=target_sizes)
        processed_results = pool.map(process_result, results)

        for max_area_box, filename in zip(processed_results, rows['filename']):
            if max_area_box is None:
                raise RuntimeError(f'No car found!!! Image: {filename}')
            boxes[filename] = max_area_box

    boxes = pd.DataFrame(boxes).T
    boxes = boxes.rename({0: 'xmin', 1: 'ymin', 2: 'xmax', 3: 'ymax'}, axis=1)
    boxes = boxes.reset_index(names='filename')

    boxes.to_csv(args.output, index=False)

