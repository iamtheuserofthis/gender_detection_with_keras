import json
import shutil
import pandas as pd
import os


def copy_prepare_sequence():
    with open('/home/gauravn/coco/annotations/annotations.json', 'r') as f:
        annotations = json.load(f)

    with open('/home/gauravn/coco/annotations/categories.json', 'r') as f:
        categories = json.load(f)

    df = pd.DataFrame(annotations)
    list_unique_id = df['image_id'].unique().tolist()

    image_dir = '/home/gauravn/val2014'
    target_dir = '/home/gauravn/non_human_from_val2014/'
    tot = len(list_unique_id)
    while len(list_unique_id) >= 1:
        image_id = list_unique_id.pop()
        print('for id :%012d, current_status: %s/%s' % (image_id, len(list_unique_id), tot))
        tmp_df = df[df['image_id'] == image_id]

        if not any(tmp_df['category_id'].values == 1):
            image_name = 'COCO_val2014_%012d.jpg' % image_id
            shutil.copy(os.path.join(image_dir, image_name), target_dir)


if __name__ == '__main__':
    copy_prepare_sequence()
