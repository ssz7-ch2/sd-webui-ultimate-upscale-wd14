from PIL import Image
import numpy as np
import pandas as pd
import cv2
import os
from pathlib import Path
from modules import shared
from modules.deepbooru import re_special as tag_escape_pattern
from huggingface_hub import hf_hub_download
from typing import List, Dict

MODEL_DIR = "wd14_tagger_model"

USE_CPU = ('all' in shared.cmd_opts.use_cpu) or (
    'interrogate' in shared.cmd_opts.use_cpu)

def postprocess_tags(
        tags: Dict[str, float],

        threshold=0.35,
        additional_tags: List[str] = [],
        exclude_tags: List[str] = [],
        sort_by_alphabetical_order=False,
        add_confident_as_weight=False,
        replace_underscore=False,
        replace_underscore_excludes: List[str] = [],
        escape_tag=False
    ) -> Dict[str, float]:
        new_tags = []
        for tag in list(tags):
            new_tag = tag

            if replace_underscore and tag not in replace_underscore_excludes:
                new_tag = new_tag.replace('_', ' ')

            if escape_tag:
                new_tag = tag_escape_pattern.sub(r'\\\1', new_tag)

            if add_confident_as_weight:
                new_tag = f'({new_tag}:{tags[tag]})'

            new_tags.append((new_tag, tags[tag]))
        tags = dict(new_tags)
        
        for t in additional_tags:
            tags[t] = 1.0

        # those lines are totally not "pythonic" but looks better to me
        tags = {
            t: c

            # sort by tag name or confident
            for t, c in sorted(
                tags.items(),
                key=lambda i: i[0 if sort_by_alphabetical_order else 1],
                reverse=not sort_by_alphabetical_order
            )

            # filter tags
            if (
                c >= threshold
                and t not in exclude_tags
            )
        }

        return tags

def preprocess_image(image, height):
    image = np.array(image)
    image = image[:, :, ::-1]

    # pad to square
    size = max(image.shape[0:2])
    pad_x = size - image.shape[1]
    pad_y = size - image.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode='constant', constant_values=255)

    interp = cv2.INTER_AREA if size > height else cv2.INTER_LANCZOS4
    image = cv2.resize(image, (height, height), interpolation=interp)

    image = image.astype(np.float32)
    return image

def get_tags(image, postprocess_opts):
    model_path = Path(hf_hub_download(repo_id='SmilingWolf/wd-v1-4-convnext-tagger-v2', filename='model.onnx', revision='v2.0', cache_dir=MODEL_DIR))
    tags_path = Path(hf_hub_download(repo_id='SmilingWolf/wd-v1-4-convnext-tagger-v2', filename='selected_tags.csv', revision='v2.0', cache_dir=MODEL_DIR))
    
    from launch import is_installed, run_pip
    if not is_installed('onnxruntime'):
        package = os.environ.get(
            'ONNXRUNTIME_PACKAGE',
            'onnxruntime-gpu'
        )

        run_pip(f'install {package}', 'onnxruntime')

    from onnxruntime import InferenceSession

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if USE_CPU:
        providers.pop(0)

    model = InferenceSession(str(model_path), providers=providers)

    tags = pd.read_csv(tags_path)

    image = image.convert('RGBA')
    new_image = Image.new('RGBA', image.size, 'WHITE')
    new_image.paste(image, mask=image)
    image = new_image.convert('RGB')

    _, height, _, _ = model.get_inputs()[0].shape

    image = preprocess_image(image, height)
    image = np.expand_dims(image, 0)

    input_name = model.get_inputs()[0].name
    label_name = model.get_outputs()[0].name
    confidents = model.run([label_name], {input_name: image})[0]

    tags = tags[:][['name']]
    tags['confidents'] = confidents[0]

    tags = dict(tags[4:].values)
    tags = postprocess_tags(tags, *postprocess_opts)
    print(', '.join(tags))
    return ', '.join(tags)