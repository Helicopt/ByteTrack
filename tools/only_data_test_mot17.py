import json
import os


"""
cd datasets
mkdir -p mix_mot20_ch/annotations
cp MOT20/annotations/val_half.json mix_mot20_ch/annotations/val_half.json
cp MOT20/annotations/test.json mix_mot20_ch/annotations/test.json
cd mix_mot20_ch
ln -s ../MOT20/train mot20_train
ln -s ../crowdhuman/CrowdHuman_train crowdhuman_train
ln -s ../crowdhuman/CrowdHuman_val crowdhuman_val
cd ..
"""

# all ids: 2215, all boxes: 1134614, all frames: 8931
# MOT20-01:: ids: 74, boxes: 19870, frames: 429

mot_json = json.load(open('datasets/MOT17/annotations/train.json', 'r'))

splits = {
    'train': ['MOT17-02','MOT17-04','MOT17-05','MOT17-09','MOT17-10','MOT17-11','MOT17-13',],
    'val': ['MOT17-04', ],
}

for part, vlist in splits.items():
    print('partition:', part)
    vlist = set(vlist)
    img_required = {}
    img_list = list()
    for img in mot_json['images']:
        img = dict(img)
        if part == 'train':
            img['file_name'] = 'mot17_train/' + img['file_name']
        video_name = img['file_name'].split('/')[-3]
        if video_name in vlist:
            img_required[img['id']] = img['file_name']
            img_list.append(img)

    ann_list = list()
    uids = set()
    max_img_id = 0
    for i, ann in enumerate(mot_json['annotations']):
        if i == 0:
            print(ann.keys(), ann['image_id'])
        max_img_id = max(max_img_id, ann['image_id'])
        if ann['image_id'] not in img_required:
            continue
        video_name = img_required[ann['image_id']].split('/')[-3]
        if i == 0:
            print(video_name)
        uids.add((ann['track_id'], video_name))
        ann_list.append(ann)

    print(mot_json['videos'])
    video_list = list(filter(lambda x: x['file_name'] in vlist, mot_json['videos']))
    category_list = mot_json['categories']

    print(list(map(len, (img_list, ann_list, video_list, category_list))))
    print(max_img_id, len(uids))

    mix_json = dict()
    mix_json['images'] = img_list
    mix_json['annotations'] = ann_list
    mix_json['videos'] = video_list
    mix_json['categories'] = category_list
    json.dump(mix_json, open('datasets/only_mot17/annotations/%s.json' % part, 'w'))