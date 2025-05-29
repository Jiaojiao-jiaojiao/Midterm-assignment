_base_ = 'mmdetection/configs/sparse_rcnn/sparse-rcnn_r50_fpn_300proposals_crop_mstrain_480-800_3x_coco.py'

dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'

classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/trainval.txt',
        img_prefix=data_root + 'VOC2007/',
        classes=classes),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/val.txt',
        img_prefix=data_root + 'VOC2007/',
        classes=classes),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        classes=classes)
)

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20)
    )
)

work_dir = './work_dirs/sparse_rcnn_r50_voc'
