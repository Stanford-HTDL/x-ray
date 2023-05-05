__author__ = "Richard Correro (richard@richardcorrero.com)"

"""
Adapted from: https://github.com/rcorrero/poisson
"""


import math
import sys
from typing import Sequence

import numpy as np
import torch
import torchvision
from torch.autograd import Variable

import mercantile


def feature(
    bbox: mercantile.LngLatBbox, fid=None, props=None, 
    projected="geographic", buffer=None, precision=None, id="Bounding Box"
):
    """Get the GeoJSON feature corresponding to a tile

    Parameters
    ----------
    bbox : LngLatBbox
    fid : str, optional
        A feature id.
    props : dict, optional
        Optional extra feature properties.
    projected : str, optional
        Non-standard web mercator GeoJSON can be created by passing
        'mercator'.
    buffer : float, optional
        Optional buffer distance for the GeoJSON polygon.
    precision : int, optional
        GeoJSON coordinates will be truncated to this number of decimal
        places.
    id : str, optional
        Name associated with geojson object.

    Returns
    -------
    dict

    """
    west, south, east, north = bbox

    if projected == "mercator":
        west, south = mercantile.xy(west, south, truncate=False)
        east, north = mercantile.xy(east, north, truncate=False)

    if buffer:
        west -= buffer
        south -= buffer
        east += buffer
        north += buffer

    if precision and precision >= 0:
        west, south, east, north = (
            round(v, precision) for v in (west, south, east, north)
        )

    bbox = [min(west, east), min(south, north), max(west, east), max(south, north)]
    geom = {
        "type": "Polygon",
        "coordinates": [
            [[west, south], [west, north], [east, north], [east, south], [west, south]]
        ],
    }

    feat = {
        "type": "Feature",
        "bbox": bbox,
        "id": id,
        "geometry": geom,
        # "properties": {"title": "XYZ tile %s" % xyz},
    }

    if props:
        feat["properties"].update(props)

    if fid is not None:
        feat["id"] = fid

    return feat


def bbox_to_geojson(
    bbox: Sequence[float], lng_lat_bbox: mercantile.LngLatBbox, 
    image_shape: Sequence[int]
) -> dict:
        y_num_pixels, x_num_pixels = image_shape # (H, W)
        west: float = lng_lat_bbox.west
        south: float = lng_lat_bbox.south
        east: float = lng_lat_bbox.east
        north: float = lng_lat_bbox.north

        if west > east:
            lng_len: float = 360.0 - (west - east) # Antemeridian problems
        else:
            lng_len: float = east - west

        lat_len: float = north - south

        x_min, y_min, x_max, y_max = bbox

        bbox_west: float = west + (x_min / x_num_pixels) * lng_len
        bbox_south: float = north  - (y_max / y_num_pixels) * lat_len
        bbox_east: float = west + (x_max / x_num_pixels) * lng_len
        bbox_north: float = north  - (y_min / y_num_pixels) * lat_len

        bbox_ll: mercantile.LngLatBbox = mercantile.LngLatBbox(
            west=bbox_west, south=bbox_south, east=bbox_east, north=bbox_north
        )

        geojson = feature(bbox_ll)
        return geojson


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, data_loader, 
    device, lr_scheduler = None, 
) -> float:
    model.train()
    running_loss = 0.0

    for (inputs, targets) in data_loader:
        inputs = [Variable(input).to(device) for input in inputs]
        targets = [{k: Variable(v).to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(inputs, targets)
        losses = sum(loss for loss in loss_dict.values())
        if not math.isfinite(losses):
            print("Loss is %-10.5f, stopping training" % losses)
            print("Loss dict:\n", loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        running_loss += losses
    return running_loss


def calculate_map(gt_boxes, pr_boxes, scores, thresh, device, form='pascal_voc'):
    def align_coordinates(boxes):
        """Align coordinates (x1,y1) < (x2,y2) to work with torchvision `box_iou` op
        Arguments:
            boxes (Tensor[N,4])
        
        Returns:
            boxes (Tensor[N,4]): aligned box coordinates
        """
        x1y1 = torch.min(boxes[:,:2,],boxes[:, 2:])
        x2y2 = torch.max(boxes[:,:2,],boxes[:, 2:])
        boxes = torch.cat([x1y1,x2y2],dim=1)
        return boxes


    def calculate_iou(gt, pr, form='pascal_voc'):
        """Calculates the Intersection over Union.
        Arguments:
            gt: (torch.Tensor[N,4]) coordinates of the ground-truth boxes
            pr: (torch.Tensor[M,4]) coordinates of the prdicted boxes
            form: (str) gt/pred coordinates format
                - pascal_voc: [xmin, ymin, xmax, ymax]
                - coco: [xmin, ymin, w, h]
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
        """
        if form == 'coco':
            gt = gt.clone()
            pr = pr.clone()

            gt[:,2] = gt[:,0] + gt[:,2]
            gt[:,3] = gt[:,1] + gt[:,3]
            pr[:,2] = pr[:,0] + pr[:,2]
            pr[:,3] = pr[:,1] + pr[:,3]

        gt = align_coordinates(gt)
        pr = align_coordinates(pr)
        
        return torchvision.ops.boxes.box_iou(gt,pr)


    def get_mappings(iou_mat):
        mappings = torch.zeros_like(iou_mat)
        _, pr_count = iou_mat.shape
        
        #first mapping (max iou for first pred_box)
        if not iou_mat[:,0].eq(0.).all():
            # if not a zero column
            mappings[iou_mat[:,0].argsort()[-1],0] = 1

        for pr_idx in range(1,pr_count):
            # Sum of all the previous mapping columns will let 
            # us know which gt-boxes are already assigned
            not_assigned = torch.logical_not(mappings[:,:pr_idx].sum(1)).long()

            # Considering unassigned gt-boxes for further evaluation 
            targets = not_assigned * iou_mat[:,pr_idx]

            # If no gt-box satisfy the previous conditions
            # for the current pred-box, ignore it (False Positive)
            if targets.eq(0).all():
                continue

            # max-iou from current column after all the filtering
            # will be the pivot element for mapping
            pivot = targets.argsort()[-1]
            mappings[pivot,pr_idx] = 1
        return mappings


    if gt_boxes.shape[0] == 0:
        if pr_boxes.shape[0] == 0:
            return 1.0
        return 0.0
    if pr_boxes.shape[0] == 0:
        return 0.0
    # sorting
    pr_boxes = pr_boxes[scores.argsort().flip(-1)]
    iou_mat = calculate_iou(gt_boxes,pr_boxes,form)
    iou_mat = iou_mat.to(device)
    
    # thresholding
    iou_mat = iou_mat.where(iou_mat > thresh, torch.tensor(0.).to(device))
    
    mappings = get_mappings(iou_mat)
    
    # mAP calculation
    tp = mappings.sum()
    fp = mappings.sum(0).eq(0).sum()
    fn = mappings.sum(1).eq(0).sum()
    mAP = tp / (tp+fp+fn)
    return mAP.cpu().detach().numpy()    


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader, device, thresh_list):
    model.eval()
    mAP_dict = {thresh: [] for thresh in thresh_list}
    for images, targets in data_loader:
        images = list(Variable(img).to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images, targets)
        # Calculate mAP
        for thresh in thresh_list:
            mAP_list = [calculate_map(target['boxes'], 
                                      output['boxes'], 
                                      output['scores'], 
                                      thresh=thresh,
                                      device=device) \
                        for target, output in zip(targets, outputs)]
            mAP_dict[thresh] += mAP_list # Creates a list of mAP's for each sample
    for thresh in thresh_list:
        mAP_dict[thresh] = np.mean(mAP_dict[thresh])
    mAP = np.mean(list(mAP_dict.values()))
    return mAP    
