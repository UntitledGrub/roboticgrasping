import argparse
import logging
import time

import numpy as np

import torch.utils.data

from utils.dataset_processing import evaluation, grasp
from utils.data import get_dataset
from models.common import post_process_output

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate TCT and FCNN')

    # Network
    parser.add_argument('--network', type=str, help='Path to saved network')
    
    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str, help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1, help='Use RGB image for evaluation (0/1)')
    parser.add_argument('--augment', action='store_true', help='Whether data augmentation should be applied')
    parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--n-grasps', type=int, default=1, help='Number of grasps to consider per image')
    parser.add_argument('--iou-eval', action='store_true', help='Compute success based on IoU metric.')
    parser.add_argument('--vis', action='store_true', help='Visualise the network output')

    args = parser.parse_args()

    return args

# --dataset jacquard --dataset-path datasets\Jacquard_Dataset --split 0.95 --iou-eval
# --dataset cornell --dataset-path datasets\Cornell_dataset --iou-eval

if __name__ == '__main__':
    args = parse_args()

    # Load Network
    net = torch.load(args.network)
    device = torch.device("cuda:0")

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    test_dataset = Dataset(args.dataset_path, ds_rotate=args.ds_rotate,
                           random_rotate=args.augment, random_zoom=args.augment,
                           include_depth=args.use_depth, include_rgb=args.use_rgb)

    indices = list(range(test_dataset.length))
    split = int(np.floor(args.split * test_dataset.length))
    
    val_indices = indices[split:]
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    logging.info('Validation size: {}'.format(len(val_indices)))

    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        sampler=val_sampler
    )
    logging.info('Done')

    results = {'correct': 0, 'failed': 0}

    total_inference_time = 0

    with torch.no_grad():
        for idx, (x, y, didx, rot, zoom) in enumerate(test_data):

            logging.info('Processing {}/{}'.format(idx+1, len(test_data)))

            start = time.time()

            xc = x.to(device)
            yc = [yi.to(device) for yi in y]
            lossd = net.compute_loss(xc, yc)

            total_inference_time += time.time() - start

            q_img, ang_img, width_img = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])

            if args.iou_eval:
                s = evaluation.calculate_iou_match(q_img, ang_img, test_data.dataset.get_gtbb(didx, rot, zoom),
                                                   no_grasps=args.n_grasps,
                                                   grasp_width=width_img,
                                                   )
                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1

            if args.vis:
                evaluation.plot_output(test_data.dataset.get_rgb(didx, rot, zoom, normalise=False),
                                        q_img, 
                                        ang_img, 
                                        width_img, 
                                        no_grasps=5, 
                                       )

    avg_inference_time = total_inference_time / len(test_data)
    logging.info('Average evaluation time per image: {}ms'.format(avg_inference_time * 1000))

    if args.iou_eval:
        logging.info('IOU Results: %d/%d = %f' % (results['correct'],
                              results['correct'] + results['failed'],
                              results['correct'] / (results['correct'] + results['failed'])))

    
