from mmdet.apis import init_detector
from mmengine.runner import load_checkpoint
from mmdet.structures.bbox import get_box_tensor
import os
import cv2
from mmengine.visualization import Visualizer
import torch

def run_inference(config_file, checkpoint_file, image_folder, output_folder, show_proposals=False):
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    visualizer = Visualizer()
    visualizer.set_dataset_meta(model.dataset_meta)

    os.makedirs(output_folder, exist_ok=True)
    image_list = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))])

    for image_name in image_list:
        image_path = os.path.join(image_folder, image_name)
        result = model.test_cfg = dict(rpn=dict(nms_pre=100), rcnn=None)
        result = model.test_step([dict(img_path=image_path)])[0]

        # Proposal 可视化
        if show_proposals and hasattr(model, 'rpn_head'):
            inputs = model.data_preprocessor([dict(img_path=image_path)], False)
            proposals = model.rpn_head.simple_test_rpn(inputs['inputs'], inputs['data_samples'])
            proposal_boxes = get_box_tensor(proposals[0].proposals)
            inputs['data_samples'][0].pred_instances = proposals[0].proposals
            visualizer.add_datasample(
                'proposal', inputs['inputs'][0], data_sample=inputs['data_samples'][0], draw_gt=False,
                show=False, wait_time=0, out_file=os.path.join(output_folder, 'proposal_' + image_name))

        # 最终预测可视化
        visualizer.add_datasample(
            image_name, result['inputs'][0], data_sample=result['data_samples'][0], draw_gt=False,
            show=False, wait_time=0, out_file=os.path.join(output_folder, image_name))

if __name__ == '__main__':
    # Mask R-CNN
    run_inference(
        config_file='mmdetection/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_voc.py',
        checkpoint_file='checkpoints/mask_rcnn.pth',
        image_folder='demo_images/',
        output_folder='results/mask_rcnn',
        show_proposals=True
    )

    # Sparse R-CNN
    run_inference(
        config_file='mmdetection/configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_voc.py',
        checkpoint_file='checkpoints/sparse_rcnn.pth',
        image_folder='demo_images/',
        output_folder='results/sparse_rcnn',
        show_proposals=True
    )
