import json
import os


"""
cd datasets
mkdir -p {target_dir}/annotations
cd {target_dir}
ln -s ../MOT20/train mot20_train
ln -s ../sompt22/train sompt_train
cd ..
"""

# f/b/v/i = 14326, 1132920, 8, 1882
# MOT20-01 = 429, 19870, 1, 74

mot_json = json.load(open('datasets/MOT20/annotations/train.json', 'r'))
sompt_json = json.load(open('datasets/sompt22/annotations/train.json', 'r'))
jsons = {
    'mot20': mot_json,
    'sompt': sompt_json,
}

# su01
target_tag = 'su01'
splits = {
    'train': {
        'mot20': ['MOT20-01', ],
        'sompt': [],
    },
    'val': {
        'mot20': ['MOT20-03', 'MOT20-05', ],
        'sompt': ['SOMPT22-11', 'SOMPT22-12', 'SOMPT22-13', ],
    }
}

# su12
target_tag = 'su12'
splits = {
    'train': {
        'mot20': ['MOT20-01', 'MOT20-02', ],
        'sompt': [],
    },
    'val': {
        'mot20': ['MOT20-03', 'MOT20-05', ],
        'sompt': ['SOMPT22-11', 'SOMPT22-12', 'SOMPT22-13', ],
    }
}

# suall
target_tag = 'suall'
splits = {
    'train': {
        'mot20': ['MOT20-01', 'MOT20-02', ],
        'sompt': ['SOMPT22-02', 'SOMPT22-04', 'SOMPT22-05', 'SOMPT22-07', 'SOMPT22-08', 'SOMPT22-10', ],
    },
    'val': {
        'mot20': ['MOT20-03', 'MOT20-05', ],
        'sompt': ['SOMPT22-11', 'SOMPT22-12', 'SOMPT22-13', ],
    }
}

# id_offset = 0
# uid_offset = 0
# video_offset = 0
# ann_offset = 0

for part in splits:
    print('partition:', part)
    img_list = list()
    ann_list = list()
    uids = set()
    video_list = list()
    category_list = list()  # mot_json['categories']
    id_offset = 0
    uid_offset = 0
    video_offset = 0
    ann_offset = 0
    max_img_id = 0
    max_track_id = 0
    max_video_id = 0
    max_ann_id = 0
    for sub, vlist in splits[part].items():
        vlist = set(vlist)
        img_required = {}
        for img in jsons[sub]['images']:
            img = dict(img)
            img['id'] += id_offset
            img['prev_image_id'] += id_offset
            img['next_image_id'] += id_offset
            img['video_id'] += video_offset
            max_img_id = max(max_img_id, img['id'])
            max_video_id = max(max_video_id, img['video_id'])
            # if part == 'train':
            img['file_name'] = '{}_train/'.format(sub) + img['file_name']
            video_name = img['file_name'].split('/')[-3]
            if video_name in vlist:
                img_required[img['id']] = img['file_name']
                img_list.append(img)
        #filter(lambda x: x['file_name'] in vlist, mot_json['videos'])
        for i, ann in enumerate(jsons[sub]['annotations']):
            ann = dict(ann)
            ann['id'] += ann_offset
            ann['image_id'] += id_offset
            ann['track_id'] += uid_offset
            if i == 0:
                print(ann.keys(), ann['image_id'])
            max_img_id = max(max_img_id, ann['image_id'])
            max_ann_id = max(max_ann_id, ann['id'])
            max_track_id = max(max_track_id, ann['track_id'])
            if ann['image_id'] not in img_required:
                continue
            video_name = img_required[ann['image_id']].split('/')[-3]
            if i == 0:
                print(video_name)
            uids.add((ann['track_id'], video_name))
            ann_list.append(ann)

        for vid in jsons[sub]['videos']:
            vid = dict(vid)
            if vid['file_name'] in vlist:
                vid['id'] += video_offset
                video_list.append(vid)

        category_list.extend(jsons[sub]['categories'])

        id_offset = max_img_id + 1
        uid_offset = max_track_id + 1
        video_offset = max_video_id + 1
        ann_offset = max_ann_id + 1

        print(jsons[sub]['videos'])
        print(jsons[sub]['categories'])
    # category_list = list(set(category_list))
    # video_list = list(filter(lambda x: x['file_name'] in vlist, mot_json['videos']))
    # category_list = mot_json['categories']
    new_cat_list = []
    cat_set = set()
    for cat in category_list:
        if cat['id'] not in cat_set:
            cat_set.add(cat['id'])
            new_cat_list.append(cat)
    category_list = new_cat_list

    print(list(map(len, (img_list, ann_list, video_list, uids, category_list))))
    print(max_img_id, max_ann_id, max_video_id, max_track_id)

    mix_json = dict()
    mix_json['images'] = img_list
    mix_json['annotations'] = ann_list
    mix_json['videos'] = video_list
    mix_json['categories'] = category_list
    target_dir = 'datasets/{}_mix20/annotations'.format(target_tag)
    os.makedirs(target_dir, exist_ok=True)
    json.dump(mix_json, open('%s/%s.json' % (target_dir, part), 'w'))
