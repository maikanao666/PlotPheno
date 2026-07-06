import os
import cv2
import numpy as np
import rasterio
import rasterio.mask
from rasterio.windows import Window
from rasterio.warp import transform_geom
import pandas as pd
from mmdet.apis import init_detector, inference_detector
import math
from shapely.geometry import Polygon, mapping
from PIL import Image
import warnings
import torch
import random
import tempfile
warnings.filterwarnings('ignore')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'models', 'st_prnet.py')
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'models', 'weight.pth')
DEVICE = 'cuda:0'
def get_bounding_box(cnt): return cv2.boundingRect(cnt)
def rotate_point(cx, cy, angle_degrees):
    angle_rad = math.radians(angle_degrees)
    rx = cx * math.cos(angle_rad) - cy * math.sin(angle_rad)
    ry = cx * math.sin(angle_rad) + cy * math.cos(angle_rad)
    return rx, ry
def pixel_to_geo(contour, transform):
    return Polygon([(transform[0] * p[0][0] + transform[1] * p[0][1] + transform[2],
                     transform[3] * p[0][0] + transform[4] * p[0][1] + transform[5]) for p in contour])
def check_overlap_and_merge_logic(item1, item2, iou_thr, ioa_thr):
    x1, y1, w1, h1 = item1['bbox']
    x2, y2, w2, h2 = item2['bbox']
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2
    if (x1 > x2_max + 5 or x2 > x1_max + 5 or y1 > y2_max + 5 or y2 > y1_max + 5): return False, 0.0
    w, h = max(x1_max, x2_max) - min(x1, x2) + 20, max(y1_max, y2_max) - min(y1, y2) + 20
    m1, m2 = np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
    ox, oy = min(x1, x2) - 10, min(y1, y2) - 10
    cv2.drawContours(m1, [item1['contour']], -1, 1, -1, offset=(-ox, -oy))
    cv2.drawContours(m2, [item2['contour']], -1, 1, -1, offset=(-ox, -oy))
    inter = np.logical_and(m1, m2).sum()
    if inter == 0: return False, 0.0
    iou = inter / np.logical_or(m1, m2).sum()
    ioa = inter / min(m1.sum(), m2.sum()) if min(m1.sum(), m2.sum()) > 0 else 0
    return (iou > iou_thr or ioa > ioa_thr), iou
def process_and_sever_watershed(raw_detections, iou_thr, ioa_thr, watershed_shrink_ratio, progress_callback):
    n = len(raw_detections)
    parent = list(range(n))
    def find(i):
        if parent[i] != i: parent[i] = find(parent[i])
        return parent[i]
    for i in range(n): raw_detections[i]['bbox'] = get_bounding_box(raw_detections[i]['contour'])
    CELL_SIZE, gap = 1000, 20
    grid = {}
    for i in range(n):
        x, y, w, h = raw_detections[i]['bbox']
        for r in range(max(0, y - gap) // CELL_SIZE, (y + h + gap) // CELL_SIZE + 1):
            for c in range(max(0, x - gap) // CELL_SIZE, (x + w + gap) // CELL_SIZE + 1):
                grid.setdefault((r, c), []).append(i)
    candidate_pairs = set()
    for indices in grid.values():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                candidate_pairs.add(tuple(sorted([indices[i], indices[j]])))
    progress_callback(30, f"合并重叠小区...")
    for (i, j) in candidate_pairs:
        if find(i) != find(j) and check_overlap_and_merge_logic(raw_detections[i], raw_detections[j], iou_thr, ioa_thr)[
            0]:
            parent[find(j)] = find(i)
    groups = {}
    for i in range(n): groups.setdefault(find(i), []).append(raw_detections[i])
    final_items = []
    progress_callback(35, "执行分水岭精细分离...")
    for items in groups.values():
        contours_list = [x['contour'] for x in items]
        min_x = min([cv2.boundingRect(c)[0] for c in contours_list])
        min_y = min([cv2.boundingRect(c)[1] for c in contours_list])
        max_x = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours_list])
        max_y = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours_list])
        pad = 50
        mask = np.zeros((max_y - min_y + pad * 2, max_x - min_x + pad * 2), dtype=np.uint8)
        off_x, off_y = min_x - pad, min_y - pad
        for cnt in contours_list: cv2.drawContours(mask, [cnt], -1, 255, -1, offset=(-off_x, -off_y))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, watershed_shrink_ratio * dist_transform.max(), 255, 0)
        sure_bg = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
        unknown = cv2.subtract(sure_bg, np.uint8(sure_fg))
        ret, markers = cv2.connectedComponents(np.uint8(sure_fg))
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), markers)
        for label in range(2, ret + 1):
            single_mask = np.zeros_like(mask)
            single_mask[markers == label] = 255
            cnts, _ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts and cv2.contourArea(max(cnts, key=cv2.contourArea)) > 30:
                final_items.append({'contour': max(cnts, key=cv2.contourArea) + [off_x, off_y]})
    return final_items
def extract_plot_data_robust(src, contour, geo_polygon, rgb_crs, rgb_w, rgb_h, custom_canopy_mask=None):
    try:
        geom_dict = mapping(geo_polygon)
        if rgb_crs is not None and src.crs is not None and rgb_crs != src.crs:
            geom_dict = transform_geom(rgb_crs, src.crs, geom_dict)
        out_img, _ = rasterio.mask.mask(src, [geom_dict], crop=True, nodata=-99999)
        out_img = out_img[0].astype(float)
        out_img[out_img == -99999] = np.nan
        if src.nodata is not None: out_img[out_img == src.nodata] = np.nan
        if custom_canopy_mask is not None:
            bx, by, bw, bh = cv2.boundingRect(contour)
            bx, by = max(0, bx), max(0, by)
            bw, bh = min(bw, rgb_w - bx), min(bh, rgb_h - by)
            if custom_canopy_mask.shape == out_img.shape:
                out_img[custom_canopy_mask == 0] = np.nan
        valid_px = out_img[~np.isnan(out_img)]
        if len(valid_px) > 0: return valid_px
    except:
        pass
    if src.width == rgb_w and src.height == rgb_h:
        try:
            bx, by, bw, bh = cv2.boundingRect(contour)
            bx, by = max(0, bx), max(0, by)
            bw, bh = min(bw, rgb_w - bx), min(bh, rgb_h - by)
            window_data = src.read(1, window=Window(bx, by, bw, bh)).astype(float)
            mask = np.zeros((bh, bw), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1, offset=(-int(bx), -int(by)))
            if custom_canopy_mask is not None:
                mask = cv2.bitwise_and(mask, custom_canopy_mask)
            valid_px = window_data[mask == 255]
            if src.nodata is not None:
                if math.isnan(src.nodata):
                    valid_px = valid_px[~np.isnan(valid_px)]
                else:
                    valid_px = valid_px[valid_px != src.nodata]
            return valid_px[~np.isnan(valid_px)]
        except:
            return None
    return None
def compute_rotation_matrix_and_sizes(w, h, angle_degrees):
    rad = math.radians(angle_degrees)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    new_w = int(abs(w * cos_a) + abs(h * sin_a))
    new_h = int(abs(w * sin_a) + abs(h * cos_a))
    M = np.zeros((2, 3), dtype=np.float32)
    M[0, 0] = cos_a;
    M[0, 1] = sin_a;
    M[0, 2] = - (w / 2.0) * cos_a - (h / 2.0) * sin_a + (new_w / 2.0)
    M[1, 0] = -sin_a;
    M[1, 1] = cos_a;
    M[1, 2] = (w / 2.0) * sin_a - (h / 2.0) * cos_a + (new_h / 2.0)
    return M, new_w, new_h
def generate_mask_only(rgb_path, output_dir, params, progress_callback):
    slice_size = int(params.get('SLICE_SIZE', 1500))
    stride = int(params.get('STRIDE', 500))
    score_thr = float(params.get('SCORE_THR', 0.65))
    merge_iou_thr = float(params.get('MERGE_IOU_THR', 0.05))
    merge_ioa_thr = float(params.get('MERGE_IOA_THR', 0.65))
    field_rotate_angle = float(params.get('FIELD_ROTATE_ANGLE', 0))
    row_tolerance = float(params.get('ROW_TOLERANCE', 400))
    font_scale = 2.2
    manual_pixel_area = float(params.get('PIXEL_AREA', 0.00002601))
    min_area_thr = 3
    min_solidity_thr = 0.45
    watershed_shrink_ratio = 0.35
    if torch.cuda.is_available(): torch.cuda.set_device(0)
    progress_callback(5, "正在初始化实例分割模型...")
    model = init_detector(CONFIG_PATH, CHECKPOINT_PATH, device=DEVICE)
    raw = []
    with rasterio.open(rgb_path) as src:
        w_img, h_img, src_transform = src.width, src.height, src.transform
        meta_area = abs(src.transform[0]) * abs(src.transform[4])
        pixel_area_m2 = meta_area if (meta_area > 1e-7 and not math.isclose(meta_area, 1.0)) else manual_pixel_area
        wins = [(x, y) for y in range(0, h_img, stride) for x in range(0, w_img, stride)]
        for idx, (x, y) in enumerate(wins):
            w_c, h_c = min(slice_size, w_img - x), min(slice_size, h_img - y)
            img_d = src.read(window=Window(x, y, w_c, h_c))
            img = cv2.cvtColor(img_d[:3].transpose(1, 2, 0), cv2.COLOR_RGB2BGR) if img_d.shape[
                                                                                       0] >= 3 else cv2.cvtColor(
                img_d[0], cv2.COLOR_GRAY2BGR)
            with torch.no_grad():
                res = inference_detector(model, img)
            if hasattr(res, 'pred_instances') and res.pred_instances.masks is not None:
                masks, scores = res.pred_instances.masks.cpu().numpy(), res.pred_instances.scores.cpu().numpy()
                for i, s in enumerate(scores):
                    if s >= score_thr:
                        cnts, _ = cv2.findContours(masks[i].astype(np.uint8), cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
                        for c in cnts:
                            if cv2.contourArea(c) > 30: raw.append({'contour': c + [x, y], 'score': s})
            del res;
            torch.cuda.empty_cache()
            progress_callback(5 + int((idx / len(wins)) * 25), f"RGB 切片推理中... ({idx + 1}/{len(wins)})")
    final = process_and_sever_watershed(raw, merge_iou_thr, merge_ioa_thr, watershed_shrink_ratio, progress_callback)
    progress_callback(40, "正在过滤小区并排序...")
    valid_items = []
    for item in final:
        cnt = item['contour']
        calc_val = cv2.contourArea(cnt) * pixel_area_m2
        if calc_val < min_area_thr: continue
        hull = cv2.convexHull(cnt)
        if (
        cv2.contourArea(cnt) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0) < min_solidity_thr: continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        cx, cy = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (bx + bw // 2, by + bh // 2)
        valid_items.append(
            {'contour': cnt, 'cx': cx, 'cy': cy, 'geometry': pixel_to_geo(cnt, src_transform), 'bx': bx, 'by': by,
             'bw': bw, 'bh': bh, 'area_m2': calc_val})
    center = (w_img / 2.0, h_img / 2.0)
    rad = math.radians(field_rotate_angle)
    new_w = int(abs(w_img * math.cos(rad)) + abs(h_img * math.sin(rad)))
    new_h = int(abs(w_img * math.sin(rad)) + abs(h_img * math.cos(rad)))
    M_affine = cv2.getRotationMatrix2D(center, -field_rotate_angle, 1.0)
    M_affine[0, 2] += (new_w / 2.0) - center[0]
    M_affine[1, 2] += (new_h / 2.0) - center[1]
    for item in valid_items:
        cx, cy = item['cx'], item['cy']
        item['rx'] = M_affine[0, 0] * cx + M_affine[0, 1] * cy + M_affine[0, 2]
        item['ry'] = M_affine[1, 0] * cx + M_affine[1, 1] * cy + M_affine[1, 2]
    sorted_by_y = sorted(valid_items, key=lambda k: k['ry'], reverse=True)
    rows = []
    for item in sorted_by_y:
        placed = False
        for r in rows:
            row_avg_y = sum(x['ry'] for x in r) / len(r)
            if abs(item['ry'] - row_avg_y) <= row_tolerance:
                r.append(item)
                placed = True
                break
        if not placed: rows.append([item])
    rows.sort(key=lambda r: sum(x['ry'] for x in r) / len(r), reverse=True)
    for r in rows: r.sort(key=lambda k: k['rx'])
    flattened_plots = []
    for r_id, row in enumerate(rows):
        for c_id, item in enumerate(row):
            item['plot_id'] = f"{r_id + 1}-{c_id + 1}"
            flattened_plots.append(item)
    progress_callback(60, "正在生成 Overlay 及 Mask 图像...")
    base_name = os.path.splitext(os.path.basename(rgb_path))[0]
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    with rasterio.open(rgb_path) as src:
        vis = cv2.cvtColor(src.read((1, 2, 3)).transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    vis_m = np.zeros((h_img, w_img, 3), np.uint8)
    Image.MAX_IMAGE_PIXELS = None
    vis_pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    rotated_pil = vis_pil.rotate(-field_rotate_angle, resample=Image.BILINEAR, expand=True, fillcolor=(0, 0, 0))
    rotated_vis = cv2.cvtColor(np.array(rotated_pil), cv2.COLOR_RGB2BGR)
    del vis_pil, vis
    thickness = max(1, int(font_scale * 2.5))
    for item in flattened_plots:
        col = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        cv2.drawContours(vis_m, [item['contour']], -1, col, -1)
        pts = item['contour'].reshape(-1, 2)
        pts_ones = np.hstack([pts, np.ones((len(pts), 1))])
        transformed_pts = M_affine.dot(pts_ones.T).T
        new_cnt = transformed_pts.reshape(-1, 1, 2).astype(np.int32)
        cv2.drawContours(rotated_vis, [new_cnt], -1, col, 15)
        M_new = cv2.moments(new_cnt)
        cx_rot = int(M_new["m10"] / M_new["m00"]) if M_new["m00"] != 0 else int(item['rx'])
        cy_rot = int(M_new["m01"] / M_new["m00"]) if M_new["m00"] != 0 else int(item['ry'])
        txt = item['plot_id']
        (t_w, t_h), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
        tx = int(cx_rot - t_w / 2)
        ty = int(cy_rot + t_h / 2)
        cv2.putText(rotated_vis, txt, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), thickness + 15)
        cv2.putText(rotated_vis, txt, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), thickness)
    vis_m_pil = Image.fromarray(vis_m)
    rotated_mask_pil = vis_m_pil.rotate(-field_rotate_angle, resample=Image.NEAREST, expand=True, fillcolor=(0, 0, 0))
    rotated_vis_m = cv2.cvtColor(np.array(rotated_mask_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(overlay_path, rotated_vis)
    cv2.imwrite(mask_path, rotated_vis_m)
    progress_callback(100, "掩码生成成功！")
    return flattened_plots, overlay_path, mask_path
def extract_selected_phenotypes(flattened_plots, rgb_path, bound_tasks, output_dir, params, progress_callback,
                                exg_rgb_path=None):
    results_for_ui = []
    total_plots = len(flattened_plots)
    manual_pixel_area = float(params.get('PIXEL_AREA', 0.00002601))
    standard_plot_area = float(params.get('STANDARD_PLOT_AREA', 13.5))
    field_rotate_angle = float(params.get('FIELD_ROTATE_ANGLE', 0))
    font_scale = float(params.get('FONT_SCALE', 1.2))
    area_source_path = bound_tasks.get('Area', rgb_path) if bound_tasks.get('Area') else rgb_path
    with rasterio.open(rgb_path) as src:
        rgb_w, rgb_h, rgb_crs = src.width, src.height, src.crs
    with rasterio.open(area_source_path) as src:
        meta_area = abs(src.transform[0]) * abs(src.transform[4])
        pixel_area_m2 = meta_area if (meta_area > 1e-7 and not math.isclose(meta_area, 1.0)) else manual_pixel_area
    target_rgb_src = rgb_path if exg_rgb_path is None else exg_rgb_path
    with rasterio.open(target_rgb_src) as src:
        full_img = cv2.cvtColor(src.read((1, 2, 3)).transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    pure_full_image = np.ones_like(full_img) * 255
    open_srcs = {}
    for task, t_path in bound_tasks.items():
        if task in ['FVC', 'Area', 'EXG', 'Volume', 'VDVI']:
            continue
        if isinstance(t_path, dict):
            open_srcs[task] = {}
            for band_name, b_path in t_path.items():
                if b_path and os.path.exists(b_path):
                    open_srcs[task][band_name] = rasterio.open(b_path)
        elif t_path and os.path.exists(t_path):
            open_srcs[task] = rasterio.open(t_path)
    if 'Volume' in bound_tasks and bound_tasks['Volume'] and os.path.exists(bound_tasks['Volume']):
        open_srcs['Volume'] = rasterio.open(bound_tasks['Volume'])
    for idx, plot in enumerate(flattened_plots):
        plot_dict = {"Plot_ID": plot['plot_id']}
        plot_dict['contour'] = plot['contour']
        plot_dict['cx'], plot_dict['cy'] = plot['cx'], plot['cy']
        plot_roi = full_img[plot['by']:plot['by'] + plot['bh'], plot['bx']:plot['bx'] + plot['bw']]
        plot_mask = np.zeros((plot['bh'], plot['bw']), np.uint8)
        cv2.drawContours(plot_mask, [plot['contour']], -1, 255, -1, offset=(-plot['bx'], -plot['by']))
        roi_blur = cv2.medianBlur(plot_roi, 3)
        b, g, r = cv2.split(roi_blur.astype(np.float32))
        exg_matrix = 2 * g - r - b
        valid_exg_values = exg_matrix[plot_mask == 255]
        if len(valid_exg_values) > 0:
            p_min = np.percentile(valid_exg_values, 40)
            p_max = np.percentile(valid_exg_values, 60)
            adaptive_threshold = (p_min + p_max) / 2.0
            if adaptive_threshold < 15.0:
                adaptive_threshold = 15.0
            elif adaptive_threshold > 22.0:
                adaptive_threshold = 22.0
            canopy_binary = ((exg_matrix > adaptive_threshold) & (g > r) & (plot_mask == 255)).astype(np.uint8) * 255
        else:
            canopy_binary = np.zeros_like(plot_mask)
        if np.any(canopy_binary == 255):
            global_roi = full_img[plot['by']:plot['by'] + plot['bh'], plot['bx']:plot['bx'] + plot['bw']]
            white_roi = pure_full_image[plot['by']:plot['by'] + plot['bh'], plot['bx']:plot['bx'] + plot['bw']]
            white_roi[canopy_binary == 255] = global_roi[canopy_binary == 255]
            pure_full_image[plot['by']:plot['by'] + plot['bh'], plot['bx']:plot['bx'] + plot['bw']] = white_roi
        if 'Area' in bound_tasks:
            plot_dict['Area'] = float(round(plot['area_m2'], 4))
        if 'FVC' in bound_tasks:
            fvc = min((np.sum(canopy_binary == 255) * pixel_area_m2 / standard_plot_area) * 100, 100.0)
            plot_dict['FVC'] = fvc
        if 'EXG' in bound_tasks:
            valid_pure_exg = exg_matrix[canopy_binary == 255]
            plot_dict['EXG'] = float(np.mean(valid_pure_exg)) if len(valid_pure_exg) > 0 else 0.0
        if 'VDVI' in bound_tasks:
            denom = 2 * g + r + b
            denom[denom == 0] = 1.0
            vdvi_matrix = (2 * g - r - b) / denom
            valid_vdvi = vdvi_matrix[canopy_binary == 255]
            plot_dict['VDVI'] = float(np.mean(valid_vdvi)) if len(valid_vdvi) > 0 else 0.0
        for task in ['NDVI', 'NDRE', 'LCI', 'GNDVI', 'OSAVI', 'EVI', 'SAVI', 'PH']:
            if task in bound_tasks and task in open_srcs:
                if task == 'SAVI':
                    nir_key = next((k for k in open_srcs[task].keys() if 'NIR' in k or '近红外' in k), None)
                    red_key = next((k for k in open_srcs[task].keys() if 'Red' in k or '红光' in k), None)
                    if nir_key and red_key:
                        nir_px = extract_plot_data_robust(open_srcs[task][nir_key], plot['contour'], plot['geometry'],
                                                          rgb_crs, rgb_w, rgb_h, custom_canopy_mask=canopy_binary)
                        red_px = extract_plot_data_robust(open_srcs[task][red_key], plot['contour'], plot['geometry'],
                                                          rgb_crs, rgb_w, rgb_h, custom_canopy_mask=canopy_binary)
                        if nir_px is not None and red_px is not None and len(nir_px) > 0 and len(nir_px) == len(red_px):
                            L = 0.5
                            savi_arr = ((nir_px - red_px) / (nir_px + red_px + L)) * (1.0 + L)
                            plot_dict[task] = float(np.mean(savi_arr))
                        else:
                            plot_dict[task] = 0.0
                    else:
                        plot_dict[task] = 0.0
                elif task == 'EVI':
                    nir_key = next((k for k in open_srcs[task].keys() if 'NIR' in k or '近红外' in k), None)
                    red_key = next((k for k in open_srcs[task].keys() if 'Red' in k or '红光' in k), None)
                    blue_key = next((k for k in open_srcs[task].keys() if 'Blue' in k or '蓝光' in k), None)
                    if nir_key and red_key and blue_key:
                        nir_px = extract_plot_data_robust(open_srcs[task][nir_key], plot['contour'], plot['geometry'],
                                                          rgb_crs, rgb_w, rgb_h, custom_canopy_mask=canopy_binary)
                        red_px = extract_plot_data_robust(open_srcs[task][red_key], plot['contour'], plot['geometry'],
                                                          rgb_crs, rgb_w, rgb_h, custom_canopy_mask=canopy_binary)
                        blue_px = extract_plot_data_robust(open_srcs[task][blue_key], plot['contour'], plot['geometry'],
                                                           rgb_crs, rgb_w, rgb_h, custom_canopy_mask=canopy_binary)
                        if nir_px is not None and red_px is not None and blue_px is not None and len(nir_px) > 0:
                            evi_arr = 2.5 * ((nir_px - red_px) / (nir_px + 6.0 * red_px - 7.5 * blue_px + 1.0))
                            plot_dict[task] = float(np.mean(evi_arr))
                        else:
                            plot_dict[task] = 0.0
                    else:
                        plot_dict[task] = 0.0
                else:
                    valid_data = extract_plot_data_robust(open_srcs[task], plot['contour'], plot['geometry'], rgb_crs,
                                                          rgb_w, rgb_h, custom_canopy_mask=canopy_binary)
                    if valid_data is not None and len(valid_data) > 0:
                        if task == 'PH':
                            valid_data = valid_data[valid_data > 0]
                            plot_dict[task] = float(np.percentile(valid_data, 95)) if len(valid_data) > 0 else 0.0
                        else:
                            plot_dict[task] = float(np.mean(valid_data))
                    else:
                        plot_dict[task] = 0.0
        if 'Volume' in bound_tasks and 'Volume' in open_srcs:
            chm_px = extract_plot_data_robust(open_srcs['Volume'], plot['contour'], plot['geometry'], rgb_crs, rgb_w,
                                              rgb_h, custom_canopy_mask=canopy_binary)
            if chm_px is not None and len(chm_px) > 0:
                chm_px = chm_px[chm_px > 0]
                plot_dict['Volume'] = float(np.sum(chm_px) * pixel_area_m2) if len(chm_px) > 0 else 0.0
            else:
                plot_dict['Volume'] = 0.0
        for key, val in plot_dict.items():
            if isinstance(val, float):
                if key in ['Area', 'Volume']:
                    plot_dict[key] = round(val, 4)
                else:
                    plot_dict[key] = round(val, 2)
        results_for_ui.append(plot_dict)
        progress_callback(int((idx + 1) / total_plots * 75),
                          f"提取纯净表型: {plot['plot_id']} ({idx + 1}/{total_plots})")
    for src_obj in open_srcs.values():
        if isinstance(src_obj, dict):
            for sub_src in src_obj.values():
                sub_src.close()
        else:
            src_obj.close()
    pure_full_path = os.path.join(output_dir, "pure_canopy_full_map.png")
    cv2.imwrite(pure_full_path, pure_full_image)
    cleaned_results = []
    for item in results_for_ui:
        d = {k: v for k, v in item.items() if k not in ['contour', 'cx', 'cy']}
        cleaned_results.append(d)
    available_metrics = [m for m in
                         ['FVC', 'Area', 'NDVI', 'NDRE', 'PH', 'EXG', 'LCI', 'GNDVI', 'OSAVI', 'EVI', 'SAVI', 'Volume',
                          'VDVI'] if m in bound_tasks]
    target_rgb_src = rgb_path if exg_rgb_path is None else exg_rgb_path
    base_name = os.path.splitext(os.path.basename(target_rgb_src))[0]
    heatmap_paths = re_render_heatmaps(results_for_ui, available_metrics, output_dir, params, progress_callback,
                                       base_name=base_name)
    return cleaned_results, heatmap_paths, results_for_ui
def re_render_heatmaps(results_for_ui, available_metrics, output_dir, params, progress_callback, base_name=""):
    progress_callback(85, "正在渲染高清热力图...")
    heatmap_paths = {}
    from PIL import ImageDraw, ImageFont
    HD_SCALE = 4
    BOX_WIDTH = int(params.get('BOX_WIDTH', 60)) * HD_SCALE
    BOX_HEIGHT = int(params.get('BOX_HEIGHT', 120)) * HD_SCALE
    BOX_GAP = int(params.get('BOX_GAP', 10)) * HD_SCALE
    MARGIN_TOP = int(params.get('MARGIN_TOP', 50)) * HD_SCALE
    MARGIN_BOTTOM = int(params.get('MARGIN_BOTTOM', 50)) * HD_SCALE
    MARGIN_LEFT = int(params.get('MARGIN_LEFT', 50)) * HD_SCALE
    MARGIN_RIGHT = int(params.get('MARGIN_RIGHT', 180)) * HD_SCALE
    TEXT_SCALE_ID = float(params.get('TEXT_SCALE_ID', 0.6)) * HD_SCALE
    TEXT_SCALE_VAL = float(params.get('TEXT_SCALE_VAL', 0.5)) * HD_SCALE
    VERTICAL_SPACING = int(params.get('VERTICAL_SPACING', 15)) * HD_SCALE
    try:
        font_path = "arial.ttf" if os.name == 'nt' else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font_id = ImageFont.truetype(font_path, int(22 * TEXT_SCALE_ID))
        font_val = ImageFont.truetype(font_path, int(20 * TEXT_SCALE_VAL))
        font_title = ImageFont.truetype(font_path, int(26 * (float(params.get('TEXT_SCALE_ID', 0.6)) * HD_SCALE)))
    except:
        font_id = ImageFont.load_default()
        font_val = ImageFont.load_default()
        font_title = ImageFont.load_default()
    LINE_THIN = max(1, int(1 * HD_SCALE * 0.5))
    LINE_THICK = max(2, int(2 * HD_SCALE * 0.5))
    CB_WIDTH = int(params.get('CB_WIDTH', 30)) * HD_SCALE
    CB_MAX_HEIGHT = int(params.get('CB_HEIGHT', 400)) * HD_SCALE
    CB_RIGHT_OFFSET = int(params.get('CB_RIGHT_OFFSET', 40)) * HD_SCALE
    UNIT_DICT = {'FVC': '(%)', 'Area': '(m2)', 'PH': '(m)', 'Volume': '(m3)'}
    for metric in available_metrics:
        vals = []
        for item in results_for_ui:
            if metric not in item:
                item[metric] = 0.0
            val = item[metric]
            if isinstance(val, (int, float, np.number)): vals.append(float(val))
        if not vals: continue
        min_v, max_v = min(vals), max(vals)
        range_v = max_v - min_v if max_v != min_v else 1.0
        max_row, max_col = 0, 0
        for item in results_for_ui:
            if 'Plot_ID' in item and '-' in item['Plot_ID']:
                r_str, c_str = item['Plot_ID'].split('-')
                max_row = max(max_row, int(r_str))
                max_col = max(max_col, int(c_str))
        if max_row == 0 or max_col == 0: continue
        img_w = MARGIN_LEFT + max_col * BOX_WIDTH + (max_col - 1) * BOX_GAP + MARGIN_RIGHT
        img_h = MARGIN_TOP + max_row * BOX_HEIGHT + (max_row - 1) * BOX_GAP + MARGIN_BOTTOM
        heatmap_img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
        for item in results_for_ui:
            val = float(item.get(metric, 0.0))
            r_idx = int(item['Plot_ID'].split('-')[0]) - 1
            c_idx = int(item['Plot_ID'].split('-')[1]) - 1
            draw_r_idx = (max_row - 1) - r_idx
            x1 = MARGIN_LEFT + c_idx * (BOX_WIDTH + BOX_GAP)
            y1 = MARGIN_TOP + draw_r_idx * (BOX_HEIGHT + BOX_GAP)
            x2 = x1 + BOX_WIDTH
            y2 = y1 + BOX_HEIGHT
            ratio = (val - min_v) / range_v
            H = int(35 + ratio * 40)
            S = int(100 + ratio * 155)
            V = int(240 - ratio * 90)
            hsv_color = np.uint8([[[H, S, V]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            col_B, col_G, col_R = int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])
            cv2.rectangle(heatmap_img, (x1, y1), (x2, y2), (col_B, col_G, col_R), -1)
            cv2.rectangle(heatmap_img, (x1, y1), (x2, y2), (200, 200, 200), LINE_THIN)
        pil_img = Image.fromarray(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        for item in results_for_ui:
            val = float(item.get(metric, 0.0))
            r_idx = int(item['Plot_ID'].split('-')[0]) - 1
            c_idx = int(item['Plot_ID'].split('-')[1]) - 1
            draw_r_idx = (max_row - 1) - r_idx
            x1 = MARGIN_LEFT + c_idx * (BOX_WIDTH + BOX_GAP)
            y1 = MARGIN_TOP + draw_r_idx * (BOX_HEIGHT + BOX_GAP)
            txt_id = item['Plot_ID']
            txt_val = f"{val:.4f}" if metric in ['Area', 'Volume'] else f"{val:.2f}"
            cx, cy = x1 + BOX_WIDTH // 2, y1 + BOX_HEIGHT // 2
            id_bbox = draw.textbbox((0, 0), txt_id, font=font_id)
            val_bbox = draw.textbbox((0, 0), txt_val, font=font_val)
            tw1, th1 = id_bbox[2] - id_bbox[0], id_bbox[3] - id_bbox[1]
            tw2, th2 = val_bbox[2] - val_bbox[0], val_bbox[3] - val_bbox[1]
            draw.text((cx - tw1 // 2, cy - th1 - (VERTICAL_SPACING // 2)), txt_id, fill=(0, 0, 0), font=font_id)
            draw.text((cx - tw2 // 2, cy + (VERTICAL_SPACING // 2)), txt_val, fill=(0, 0, 0), font=font_val)
        heatmap_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cb_height = int(img_h * 0.5)
        if cb_height > CB_MAX_HEIGHT: cb_height = CB_MAX_HEIGHT
        cb_x = img_w - MARGIN_RIGHT + CB_RIGHT_OFFSET
        cb_y = img_h - MARGIN_BOTTOM - cb_height
        for y in range(cb_height):
            ratio = 1.0 - (y / cb_height)
            H = int(35 + ratio * 40)
            S = int(100 + ratio * 155)
            V = int(240 - ratio * 90)
            hsv_color = np.uint8([[[H, S, V]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            col_B, col_G, col_R = int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])
            cv2.line(heatmap_img, (cb_x, cb_y + y), (cb_x + CB_WIDTH, cb_y + y), (col_B, col_G, col_R), LINE_THIN)
        cv2.rectangle(heatmap_img, (cb_x, cb_y), (cb_x + CB_WIDTH, cb_y + cb_height), (0, 0, 0), LINE_THICK)
        pil_img = Image.fromarray(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        step_val = range_v / 4.0
        for idx in range(5):
            cur_val = min_v + idx * step_val
            lbl_y = cb_y + cb_height - int((idx / 4) * cb_height)
            cv2.line(heatmap_img, (cb_x + CB_WIDTH, lbl_y), (cb_x + CB_WIDTH + 8 * HD_SCALE, lbl_y), (0, 0, 0),
                     LINE_THICK)
        pil_img = Image.fromarray(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        for idx in range(5):
            cur_val = min_v + idx * step_val
            lbl_y = cb_y + cb_height - int((idx / 4) * cb_height)
            lbl_txt = f"{cur_val:.2f}"
            val_bbox = draw.textbbox((0, 0), lbl_txt, font=font_val)
            th = val_bbox[3] - val_bbox[1]
            draw.text((cb_x + CB_WIDTH + 15 * HD_SCALE, lbl_y - th // 2), lbl_txt, fill=(0, 0, 0), font=font_val)
        display_title = f"{metric} {UNIT_DICT.get(metric, '')}"
        title_bbox = draw.textbbox((0, 0), display_title, font=font_title)
        th_t = title_bbox[3] - title_bbox[1]
        draw.text((cb_x - 10 * HD_SCALE, cb_y - th_t - int(10 * HD_SCALE)), display_title, fill=(0, 0, 0),
                  font=font_title)
        heatmap_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        prefix = f"{base_name}_" if base_name else ""
        h_path = os.path.join(output_dir, f"{prefix}{metric}.png")
        cv2.imwrite(h_path, heatmap_img)
        heatmap_paths[metric] = (h_path, min_v, max_v)
    return heatmap_paths
def align_multitemporal_sheets(file_paths, trait_cols):
    aligned_df = None
    for epoch_name, path in file_paths.items():
        if not path or not os.path.exists(path):
            continue
        try:
            df = pd.read_excel(path)
            df['Plot_ID'] = df['Plot_ID'].astype(str).str.strip()
            keep_cols = ['Plot_ID'] + [c for c in trait_cols if c in df.columns]
            df_subset = df[keep_cols].copy()
            rename_dict = {c: f"{c}_{epoch_name}" for c in df_subset.columns if c != 'Plot_ID'}
            df_subset.rename(columns=rename_dict, inplace=True)
            if aligned_df is None:
                aligned_df = df_subset
            else:
                aligned_df = pd.merge(aligned_df, df_subset, on='Plot_ID', how='inner')
        except Exception as e:
            print(f"解析文件 {path} 失败: {str(e)}")
    return aligned_df
def calculate_growth_kinetics(aligned_df, dt1, dt2, jointing_epoch, heading_epoch, filling_epoch):
    if aligned_df is None or aligned_df.empty:
        return aligned_df
    ph_jointing_col = f"PH_{jointing_epoch}"
    ph_heading_col = f"PH_{heading_epoch}"
    ndre_heading_col = f"NDRE_{heading_epoch}"
    ndre_filling_col = f"NDRE_{filling_epoch}"
    if ph_jointing_col in aligned_df.columns and ph_heading_col in aligned_df.columns:
        aligned_df['V_height'] = (aligned_df[ph_heading_col] - aligned_df[ph_jointing_col]) / float(dt1)
        aligned_df['V_height'] = aligned_df['V_height'].round(4)
    else:
        aligned_df['V_height'] = np.nan
    if ndre_heading_col in aligned_df.columns and ndre_filling_col in aligned_df.columns:
        aligned_df['SenescenceRate'] = (aligned_df[ndre_filling_col] - aligned_df[ndre_heading_col]) / float(dt2)
        aligned_df['SenescenceRate'] = aligned_df['SenescenceRate'].round(4)
    else:
        aligned_df['SenescenceRate'] = np.nan
    return aligned_df
def parse_planting_map(file_path):
    if not file_path or not os.path.exists(file_path):
        return {}
    try:
        df = pd.read_excel(file_path, index_col=0)
        mapping_dict = {}
        for row_label, row_data in df.iterrows():
            for col_label, val in row_data.items():
                if pd.notna(val):
                    try:
                        r = int(float(row_label))
                        c = int(float(col_label))
                        plot_id = f"{r}-{c}"
                        mapping_dict[plot_id] = str(val).strip()
                    except ValueError:
                        plot_id = f"{str(row_label).strip()}-{str(col_label).strip()}"
                        mapping_dict[plot_id] = str(val).strip()
        return mapping_dict
    except Exception as e:
        print(f"解析种植图失败: {str(e)}")
        return {}
def calculate_dynamic_kinetics_pipeline(file_paths, trait_col, map_path, interval_days_list):
    import numpy as np
    import pandas as pd
    import os
    try:
        if map_path.endswith('.csv'):
            map_df = pd.read_csv(map_path, header=None)
        else:
            map_df = pd.read_excel(map_path, header=None)
    except Exception as e:
        print(f"种植图读取失败: {e}")
        return None, None
    map_vals = map_df.values
    map_list = []
    first_cell = str(map_vals[0][0]).strip()
    has_header_map = (
                'row' in first_cell.lower() or 'col' in first_cell.lower() or '\\' in first_cell or '/' in first_cell)
    if has_header_map:
        for r_idx in range(1, len(map_vals)):
            try:
                row_num = int(float(str(map_vals[r_idx][0]).strip()))
            except:
                continue
            for c_idx in range(1, len(map_vals[0])):
                try:
                    col_num = int(float(str(map_vals[0][c_idx]).strip()))
                    variety = str(map_vals[r_idx][c_idx]).strip()
                    if variety == 'nan' or not variety or variety == '':
                        continue
                    plot_id = f"{row_num}-{col_num}"
                    map_list.append({'Plot_ID': plot_id, 'Variety': variety})
                except:
                    continue
    else:
        total_rows = len(map_vals)
        for r_idx in range(total_rows):
            row_num = total_rows - r_idx
            for c_idx in range(len(map_vals[0])):
                col_num = c_idx + 1
                variety = str(map_vals[r_idx][c_idx]).strip()
                if variety == 'nan' or not variety or variety == '':
                    continue
                plot_id = f"{row_num}-{col_num}"
                map_list.append({'Plot_ID': plot_id, 'Variety': variety})
    if not map_list:
        print("种植图解析失败或为空")
        return None, None
    mapping_base_df = pd.DataFrame(map_list)
    aligned_df = mapping_base_df.copy()
    sorted_epochs = list(file_paths.keys())
    for stage in sorted_epochs:
        ep_col = f"{trait_col}_{stage}"
        try:
            path = file_paths[stage]
            if not path or not os.path.exists(path):
                if ep_col not in aligned_df.columns:
                    aligned_df[ep_col] = np.nan
                continue
            if path.endswith('.csv'):
                stage_df = pd.read_csv(path, header=None)
            else:
                stage_df = pd.read_excel(path, header=None)
            matrix_vals = stage_df.values
            tmp_list = []
            stage_first_cell = str(matrix_vals[0][0]).strip()
            has_header_stage = (
                        'row' in stage_first_cell.lower() or 'col' in stage_first_cell.lower() or '\\' in stage_first_cell or '/' in stage_first_cell)
            if has_header_stage:
                for r_idx in range(1, len(matrix_vals)):
                    try:
                        r_str = str(matrix_vals[r_idx][0]).split('.')[0].strip()
                        r_num = int(r_str)
                    except:
                        continue
                    for c_idx in range(1, len(matrix_vals[0])):
                        try:
                            c_str = str(matrix_vals[0][c_idx]).split('.')[0].strip()
                            c_num = int(c_str)
                            val = float(matrix_vals[r_idx][c_idx])
                            tmp_list.append({'Plot_ID': f"{r_num}-{c_num}", ep_col: val})
                        except:
                            continue
            else:
                st_rows = len(matrix_vals)
                for r_idx in range(st_rows):
                    r_num = st_rows - r_idx
                    for c_idx in range(len(matrix_vals[0])):
                        try:
                            c_num = c_idx + 1
                            val = float(matrix_vals[r_idx][c_idx])
                            tmp_list.append({'Plot_ID': f"{r_num}-{c_num}", ep_col: val})
                        except:
                            continue
            if tmp_list:
                stage_data_clean = pd.DataFrame(tmp_list)
                aligned_df = pd.merge(aligned_df, stage_data_clean, on='Plot_ID', how='left')
        except Exception as e:
            print(f"解析异常: {e}")
            pass
        if ep_col not in aligned_df.columns:
            aligned_df[ep_col] = np.nan
    total_days = sum(interval_days_list) if interval_days_list else 1.0
    if total_days <= 0:
        total_days = 1.0
    first_col = f"{trait_col}_{sorted_epochs[0]}"
    last_col = f"{trait_col}_{sorted_epochs[-1]}"
    aligned_df['Calculated_Rate'] = np.nan
    if first_col in aligned_df.columns and last_col in aligned_df.columns:
        aligned_df[first_col] = pd.to_numeric(aligned_df[first_col], errors='coerce')
        aligned_df[last_col] = pd.to_numeric(aligned_df[last_col], errors='coerce')
        aligned_df['Calculated_Rate'] = (aligned_df[last_col] - aligned_df[first_col]) / float(total_days)
    cols_order = ['Plot_ID', 'Variety'] + [f"{trait_col}_{ep}" for ep in sorted_epochs if f"{trait_col}_{ep}" in aligned_df.columns]
    plot_level_df = aligned_df[cols_order].copy()
    num_cols = [c for c in plot_level_df.columns if c not in ['Plot_ID', 'Variety']]
    for col in num_cols:
        plot_level_df[col] = pd.to_numeric(plot_level_df[col], errors='coerce')
    def safe_plot_join(x):
        return ", ".join([str(v).split('.')[0].strip() for v in x if pd.notna(v)])
    agg_funcs = {'Plot_ID': safe_plot_join}
    for ep in sorted_epochs:
        ep_col = f"{trait_col}_{ep}"
        if ep_col in plot_level_df.columns:
            agg_funcs[ep_col] = 'mean'
    variety_level_df = plot_level_df.groupby('Variety', as_index=False).agg(agg_funcs)
    v_cols = ['Variety', 'Plot_ID'] + [f"{trait_col}_{ep}" for ep in sorted_epochs if f"{trait_col}_{ep}" in variety_level_df.columns]
    variety_level_df = variety_level_df[v_cols]
    for col in num_cols:
        plot_level_df[col] = plot_level_df[col].fillna(0.0).round(2)
        if col in variety_level_df.columns:
            variety_level_df[col] = variety_level_df[col].fillna(0.0).round(2)
    return plot_level_df, variety_level_df