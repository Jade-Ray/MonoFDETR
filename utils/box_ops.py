# ------------------------------------------------------------------------
# Mono3DVG (https://github.com/ZhanYang-nwpu/Mono3DVG)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
import numpy as np
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    if isinstance(x, torch.Tensor):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)
    elif isinstance(x, np.ndarray):
        x_c, y_c, w, h = np.split(x, 4, axis=-1)
        b = np.concatenate([(x_c - 0.5 * w), (y_c - 0.5 * h),
                            (x_c + 0.5 * w), (y_c + 0.5 * h)], axis=-1)
        return b
    else:
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")


def box_cxcylrtb_to_xyxy(x):
    if isinstance(x, torch.Tensor):
        x_c, y_c, l, r, t, b = x.unbind(-1)
        bb = [(x_c - l), (y_c - t),
              (x_c + r), (y_c + b)]
        return torch.stack(bb, dim=-1)
    elif isinstance(x, np.ndarray):
        x_c, y_c, l, r, t, b = np.split(x, 6, axis=-1)
        bb = np.concatenate([(x_c - l), (y_c - t),
                             (x_c + r), (y_c + b)], axis=-1)
        return bb
    else:
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")


def box_xyxy_to_cxcywh(x):
    if isinstance(x, torch.Tensor):
        x0, y0, x1, y1 = x.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
             (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)
    elif isinstance(x, np.ndarray):
        x0, y0, x1, y1 = np.split(x, 4, axis=-1)
        b = np.concatenate([(x0 + x1) / 2, (y0 + y1) / 2,
                            (x1 - x0), (y1 - y0)], axis=-1)
        return b
    else:
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
