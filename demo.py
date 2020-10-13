import sys

sys.path.append('core')
import argparse
import glob
import os
import os.path as osp

import cv2
import mmcv
import numpy as np
import torch
from PIL import Image
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()
    cv2.imwrite('test.png', flo)


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    def run_once(img1, img2):
        img1 = torch.from_numpy(img1).permute(
            2, 0, 1).float().unsqueeze(0).to(DEVICE)
        img2 = torch.from_numpy(img2).permute(
            2, 0, 1).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            padder = InputPadder(img1.shape)
            img1, img2 = padder.pad(img1, img2)

            flow_low, flow_up = model(img1, img2, iters=20, test_mode=True)
        return flow_up

    import cv2
    data_root = 'high_res_motion_transfer/256_testdata/celeba_00000000'
    ref_path = 'high_res_motion_transfer/celeba_00000000_256.png'
    ref_path_512 = 'high_res_motion_transfer/celeba_00000000_512.png'

    save_flow_folder = 'results/flow'
    save_warped_folder = 'results/warp'
    save_warped_folder_512 = 'results/warp_512'
    mmcv.utils.mkdir_or_exist(save_flow_folder)
    mmcv.utils.mkdir_or_exist(save_warped_folder)
    mmcv.utils.mkdir_or_exist(save_warped_folder_512)

    img_ref = cv2.imread(ref_path)
    img_ref_512 = cv2.imread(ref_path_512)
    # whether bicubic upsample
    # h, w, _ = img_ref.shape
    # img_ref = cv2.resize(
    #     img_ref, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)

    img_paths = sorted(glob.glob(os.path.join(data_root, '*')))
    for idx, img_path in enumerate(img_paths):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        print(idx, basename)
        # read image
        img_input = cv2.imread(img_path)
        # whether bicubic upsample
        # h, w, _ = img_input.shape
        # img_input = cv2.resize(
        #     img_input, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        # caulcate flow from img_input (e.g., 049) to img_ref (050)
        flow_tensor = run_once(img_input, img_ref)  # [2, h, w]

        # downsample flow
        # c, h, w = flow_tensor.size()
        # flow_tensor = torch.nn.functional.interpolate(
        #     input=flow_tensor.unsqueeze_(0),
        #     size=(h // 4, w // 4),
        #     mode='bicubic',
        #     align_corners=False)
        # flow_tensor = flow_tensor / 4
        # flow_tensor.squeeze_(0)

        # save flow
        # tensor to numpy
        flow_np = flow_tensor.squeeze(0).cpu().numpy().transpose(
            1, 2, 0)  # [h, w, 2]
        flow_np = np.ascontiguousarray(flow_np, dtype=np.float32)
        flow_vis = mmcv.visualization.optflow.flow2rgb(flow_np)
        mmcv.imwrite(flow_vis * 255,
                     osp.join(save_flow_folder, f'{basename}_flow.png'))
        # warp
        warped_img = mmcv.video.flow_warp(
            img_ref, flow_np, filling_value=0, interpolate_mode='bilinear')
        # save warped images
        mmcv.imwrite(warped_img,
                     osp.join(save_warped_folder, f'{basename}_warped.png'))

        c, h, w = flow_tensor.size()
        flow_tensor = torch.nn.functional.interpolate(
            input=flow_tensor.unsqueeze_(0),
            size=(h * 2, w * 2),
            mode='bicubic',
            align_corners=False)
        flow_tensor = flow_tensor * 2
        flow_tensor.squeeze_(0)
        flow_np = flow_tensor.numpy().transpose(1, 2, 0)  # [h, w, 2]
        flow_np = np.ascontiguousarray(flow_np, dtype=np.float32)
        # warp
        warped_img_512 = mmcv.video.flow_warp(
            img_ref_512, flow_np, filling_value=0, interpolate_mode='bilinear')
        # save warped images
        mmcv.imwrite(
            warped_img_512,
            osp.join(save_warped_folder_512, f'{basename}_warped.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument(
        '--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument(
        '--alternate_corr',
        action='store_true',
        help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
