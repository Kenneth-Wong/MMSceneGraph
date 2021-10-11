import torch
import numpy as np
from skimage import measure
import mmcv
import cv2

def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4) in <x1, y1, x2, y2> format.
        bboxes2 (Tensor): shape (n, 4) in <x1, y1, x2, y2> format.
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> bbox_overlaps(bboxes1, bboxes2)
        tensor([[0.5238, 0.0500, 0.0041],
                [0.0323, 0.0452, 1.0000],
                [0.0000, 0.0000, 0.0000]])

    Example:
        >>> empty = torch.FloatTensor([])
        >>> nonempty = torch.FloatTensor([
        >>>     [0, 0, 10, 9],
        >>> ])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof']

    if isinstance(bboxes1, np.ndarray):
        bboxes1 = torch.from_numpy(bboxes1.copy())
    if isinstance(bboxes2, np.ndarray):
        bboxes2 = torch.from_numpy(bboxes2.copy())

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
    binary_mask: a 2D binary numpy array where '1's represent the object
    tolerance: Maximum distance from original points of polygon to approximated
    polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """

    polygons = []
    if isinstance(binary_mask, torch.Tensor):
        binary_mask = binary_mask.cpu().numpy()
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)  # x, y
        polygon = np.maximum(contour, 0)
        #segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        #segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(polygon)

    return polygons


def point_extractor_by_curvature(points_list, top_n=0.5):
    result_points_list = []
    eps = 1e-5
    if len(points_list) == 0:
        return np.empty(shape=(0, 2), dtype=np.float64)
    for points in points_list:
        x = points[:, 0]
        y = points[:, 1]
        dx = np.hstack((x[1:] - x[:-1], np.array(x[0] - x[-1])))
        dy = np.hstack((y[1:] - y[:-1], np.array(y[0] - y[-1])))

        ddx = np.hstack((dx[1:] - dx[:-1], np.array(dx[0] - dx[-1])))
        ddy = np.hstack((dy[1:] - dy[:-1], np.array(dy[0] - dy[-1])))
        K = (dx * ddy - dy * ddx) / (np.power((dx ** 2 + dy ** 2), 1.5) + eps)
        sort_idx = np.argsort(np.abs(K))[::-1]
        sort_points = points[sort_idx, :]

        if isinstance(top_n, int):
            result_points_list.append(sort_points[:top_n, :])
        elif isinstance(top_n, float):
            result_points_list.append(sort_points[:int(len(sort_points) * top_n), :])
    result_points = np.vstack(result_points_list)
    return result_points

def mask_to_poly(mask):
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:
            polygons.append(contour)
    return polygons


def get_point_from_mask(masks, bboxes, mask_size=56, sample_num=729, dist_sample_thr=1):
    """
    Adapted from Dense Reppoints: use distance transform sampling.
    :param masks: list[Tensor] or list[np.array]
    :param bboxes: list[Tensor] or list[np.array]
    :return:
    """
    points = []
    for mask, bbox in zip(masks, bboxes):
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.cpu().numpy()

        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1 + 1, 1)
        h = np.maximum(y2 - y1 + 1, 1)
        if np.sum(mask[y1:y1+h, x1:x1+w]) < 10: # the mask is too small, even is failed to be computed:
            # directly sample the points from the bbox
            new_mask = np.zeros(mask.shape).astype(np.uint8)
            new_mask[y1+2:y1+h-2, x1+2:x1+w-2] = 1
            mask = new_mask

        mask = mmcv.imresize(mask[y1:y1 + h, x1:x1 + w], (mask_size, mask_size))

        polygons = mask_to_poly(mask)
        distance_map = np.ones(mask.shape).astype(np.uint8)
        for poly in polygons:
            poly = np.array(poly).astype(np.int)
            for j in range(len(poly) // 2):
                x_0, y_0 = poly[2 * j:2 * j + 2]
                if j == len(poly) // 2 - 1:
                    x_1, y_1 = poly[0:2]
                else:
                    x_1, y_1 = poly[2 * j + 2:2 * j + 4]
                cv2.line(distance_map, (x_0, y_0), (x_1, y_1), 0, thickness=2)
        roi_dist_map = cv2.distanceTransform(distance_map, cv2.DIST_L2, 3)
        con_index = np.stack(np.nonzero(roi_dist_map == 0)[::-1], axis=-1)
        roi_dist_map[roi_dist_map == 0] = 1
        roi_dist_map[roi_dist_map > dist_sample_thr] = 0

        index_y, index_x = np.nonzero(roi_dist_map > 0)
        index = np.stack([index_x, index_y], axis=-1)
        _len = index.shape[0]
        if len(con_index) == 0:
            pts = np.zeros([2 * sample_num])
        else:
            repeat = sample_num // _len
            mod = sample_num % _len
            perm = np.random.choice(_len, mod, replace=False)
            draw = [index.copy() for i in range(repeat)]
            draw.append(index[perm])
            draw = np.concatenate(draw, 0)
            draw = np.random.permutation(draw)
            draw = draw + np.random.rand(*draw.shape)
            x_scale = float(w) / mask_size
            y_scale = float(h) / mask_size
            draw[:, 0] = draw[:, 0] * x_scale + x1
            draw[:, 1] = draw[:, 1] * y_scale + y1
            pts = draw.reshape(2 * sample_num)
        points.append(pts)
    return points

    #box_contour_points = binary_mask_to_polygon(mask_np)
    #filtered_points = point_extractor_by_curvature(box_contour_points, sample_points_thr)
    #filtered_points = torch.from_numpy(filtered_points).to(mask)
    # filtered_points = [torch.from_numpy(p).to(mask) for p in filtered_points]
    #points.append(filtered_points)
    #return points
