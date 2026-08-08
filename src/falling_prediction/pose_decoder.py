"""
Copyright (C) 2020-2021 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import numpy as np

COCO_FROM_OPENPOSE = np.array(
    (0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10)
)


def _convert_openpose_to_coco(pose):
    """Convert one OpenPose-18 pose to the COCO-17 ordering."""
    return pose[COCO_FROM_OPENPOSE]


class OpenPoseDecoder:
    """The Open Model Zoo OpenPose decoder, adapted to NumPy-only inference."""

    BODY_PARTS_KPT_IDS = (
        (1, 2),
        (1, 5),
        (2, 3),
        (3, 4),
        (5, 6),
        (6, 7),
        (1, 8),
        (8, 9),
        (9, 10),
        (1, 11),
        (11, 12),
        (12, 13),
        (1, 0),
        (0, 14),
        (14, 16),
        (0, 15),
        (15, 17),
        (2, 16),
        (5, 17),
    )
    BODY_PARTS_PAF_IDS = (
        12,
        20,
        14,
        16,
        22,
        24,
        0,
        2,
        4,
        6,
        8,
        10,
        28,
        30,
        34,
        32,
        36,
        18,
        26,
    )

    def __init__(
        self,
        num_joints=18,
        skeleton=BODY_PARTS_KPT_IDS,
        paf_indices=BODY_PARTS_PAF_IDS,
        max_points=100,
        score_threshold=0.1,
        min_paf_alignment_score=0.05,
        delta=0.5,
    ):
        self.num_joints = num_joints
        self.skeleton = skeleton
        self.paf_indices = paf_indices
        self.max_points = max_points
        self.score_threshold = score_threshold
        self.min_paf_alignment_score = min_paf_alignment_score
        self.delta = delta
        self.points_per_limb = 10
        self.grid = np.arange(self.points_per_limb, dtype=np.float32).reshape(1, -1, 1)

    def __call__(self, heatmaps, nms_heatmaps, pafs):
        if (
            np.ndim(heatmaps) != 4
            or np.shape(heatmaps) != (1, 19, 32, 57)
            or np.shape(nms_heatmaps) != np.shape(heatmaps)
            or np.shape(pafs) != (1, 38, 32, 57)
        ):
            raise ValueError(
                "expected heatmaps and NMS [1,19,32,57], PAFs [1,38,32,57]"
            )
        batch_size, _, h, w = heatmaps.shape
        assert batch_size == 1, "Batch size of 1 only supported"

        keypoints = self.extract_points(heatmaps, nms_heatmaps)
        pafs = np.transpose(pafs, (0, 2, 3, 1))
        if self.delta > 0:
            for kpts in keypoints:
                kpts[:, :2] += self.delta
                np.clip(kpts[:, 0], 0, w - 1, out=kpts[:, 0])
                np.clip(kpts[:, 1], 0, h - 1, out=kpts[:, 1])

        pose_entries, keypoints = self.group_keypoints(
            keypoints, pafs, pose_entry_size=self.num_joints + 2
        )
        poses, scores = self.convert_to_coco_format(pose_entries, keypoints)
        if len(poses) > 0:
            poses = np.asarray(poses, dtype=np.float32).reshape((len(poses), -1, 3))
            poses[:, :, 0] /= w
            poses[:, :, 1] /= h
        else:
            poses = np.empty((0, 17, 3), dtype=np.float32)
            scores = np.empty(0, dtype=np.float32)
        return poses, scores

    def extract_points(self, heatmaps, nms_heatmaps):
        batch_size, channels_num, _, _ = heatmaps.shape
        assert batch_size == 1, "Batch size of 1 only supported"
        assert channels_num >= self.num_joints
        xs, ys, scores = self.top_k(nms_heatmaps)
        masks = scores > self.score_threshold
        all_keypoints, keypoint_id = [], 0
        for k in range(self.num_joints):
            mask = masks[0, k]
            x, y, score = (
                xs[0, k][mask].ravel(),
                ys[0, k][mask].ravel(),
                scores[0, k][mask].ravel(),
            )
            n = len(x)
            if n == 0:
                all_keypoints.append(np.empty((0, 4), dtype=np.float32))
                continue
            x, y = self.refine(heatmaps[0, k], x, y)
            np.clip(x, 0, heatmaps.shape[3] - 1, out=x)
            np.clip(y, 0, heatmaps.shape[2] - 1, out=y)
            points = np.empty((n, 4), dtype=np.float32)
            points[:, :3] = np.column_stack((x, y, score))
            points[:, 3] = np.arange(keypoint_id, keypoint_id + n)
            keypoint_id += n
            all_keypoints.append(points)
        return all_keypoints

    def top_k(self, heatmaps):
        n, k, _, width = heatmaps.shape
        flat = heatmaps.reshape(n, k, -1)
        count = min(self.max_points, flat.shape[2])
        ind = flat.argpartition(-count, axis=2)[:, :, -count:]
        scores = np.take_along_axis(flat, ind, axis=2)
        order = np.argsort(-scores, axis=2)
        ind, scores = (
            np.take_along_axis(ind, order, axis=2),
            np.take_along_axis(scores, order, axis=2),
        )
        y, x = np.divmod(ind, width)
        return x, y, scores

    @staticmethod
    def refine(heatmap, x, y):
        h, w = heatmap.shape[-2:]
        valid = (x > 0) & (x < w - 1) & (y > 0) & (y < h - 1)
        xx, yy = x[valid], y[valid]
        x, y = x.astype(np.float32), y.astype(np.float32)
        x[valid] += np.sign(heatmap[yy, xx + 1] - heatmap[yy, xx - 1]) * 0.25
        y[valid] += np.sign(heatmap[yy + 1, xx] - heatmap[yy - 1, xx]) * 0.25
        return x, y

    @staticmethod
    def is_disjoint(pose_a, pose_b):
        return np.all(
            np.logical_or.reduce(
                (pose_a[:-2] == pose_b[:-2], pose_a[:-2] < 0, pose_b[:-2] < 0)
            )
        )

    def update_poses(
        self,
        kpt_a_id,
        kpt_b_id,
        all_keypoints,
        connections,
        pose_entries,
        pose_entry_size,
    ):
        for connection in connections:
            pose_a_idx = pose_b_idx = -1
            for j, pose in enumerate(pose_entries):
                if pose[kpt_a_id] == connection[0]:
                    pose_a_idx = j
                if pose[kpt_b_id] == connection[1]:
                    pose_b_idx = j
            if pose_a_idx < 0 and pose_b_idx < 0:
                pose = np.full(pose_entry_size, -1, dtype=np.float32)
                pose[kpt_a_id], pose[kpt_b_id] = connection[:2]
                pose[-1], pose[-2] = (
                    2,
                    np.sum(all_keypoints[list(connection[:2]), 2]) + connection[2],
                )
                pose_entries.append(pose)
            elif pose_a_idx >= 0 and pose_b_idx >= 0 and pose_a_idx != pose_b_idx:
                a, b = pose_entries[pose_a_idx], pose_entries[pose_b_idx]
                if self.is_disjoint(a, b):
                    a += b
                    a[:-2] += 1
                    a[-2] += connection[2]
                    del pose_entries[pose_b_idx]
            elif pose_a_idx >= 0 and pose_b_idx >= 0:
                pose_entries[pose_a_idx][-2] += connection[2]
            elif pose_a_idx >= 0:
                pose = pose_entries[pose_a_idx]
                if pose[kpt_b_id] < 0:
                    pose[-2] += all_keypoints[connection[1], 2]
                pose[kpt_b_id], pose[-2], pose[-1] = (
                    connection[1],
                    pose[-2] + connection[2],
                    pose[-1] + 1,
                )
            else:
                pose = pose_entries[pose_b_idx]
                if pose[kpt_a_id] < 0:
                    pose[-2] += all_keypoints[connection[0], 2]
                pose[kpt_a_id], pose[-2], pose[-1] = (
                    connection[0],
                    pose[-2] + connection[2],
                    pose[-1] + 1,
                )
        return pose_entries

    @staticmethod
    def connections_nms(a_idx, b_idx, affinity_scores):
        order = affinity_scores.argsort()[::-1]
        a_idx, b_idx, affinity_scores = (
            a_idx[order],
            b_idx[order],
            affinity_scores[order],
        )
        selected, used_a, used_b = [], set(), set()
        for t, (i, j) in enumerate(zip(a_idx, b_idx)):
            if i not in used_a and j not in used_b:
                selected.append(t)
                used_a.add(i)
                used_b.add(j)
        selected = np.asarray(selected, dtype=np.int32)
        return a_idx[selected], b_idx[selected], affinity_scores[selected]

    def group_keypoints(self, all_keypoints_by_type, pafs, pose_entry_size=20):
        all_keypoints = np.concatenate(all_keypoints_by_type, axis=0)
        pose_entries = []
        for part_id, paf_channel in enumerate(self.paf_indices):
            a_id, b_id = self.skeleton[part_id]
            ka, kb = all_keypoints_by_type[a_id], all_keypoints_by_type[b_id]
            n, m = len(ka), len(kb)
            if n == 0 or m == 0:
                continue
            a = np.broadcast_to(ka[None, :, :2], (m, n, 2))
            vec_raw = (kb[:, None, :2] - a).reshape(-1, 1, 2)
            points = (
                (vec_raw / (self.points_per_limb - 1) * self.grid + a.reshape(-1, 1, 2))
                .round()
                .astype(np.int32)
            )
            field = pafs[
                0,
                points[..., 1].ravel(),
                points[..., 0].ravel(),
                paf_channel : paf_channel + 2,
            ].reshape(-1, self.points_per_limb, 2)
            vec = vec_raw / (np.linalg.norm(vec_raw, axis=-1, keepdims=True) + 1e-6)
            affinity = (field * vec).sum(-1)
            valid = affinity > self.min_paf_alignment_score
            count = valid.sum(1)
            affinity = (affinity * valid).sum(1) / (count + 1e-6)
            valid_limbs = np.where(
                (affinity > 0) & (count / self.points_per_limb > 0.8)
            )[0]
            if len(valid_limbs) == 0:
                continue
            b_idx, a_idx = np.divmod(valid_limbs, n)
            affinity = affinity[valid_limbs]
            a_idx, b_idx, affinity = self.connections_nms(a_idx, b_idx, affinity)
            connections = list(
                zip(
                    ka[a_idx, 3].astype(np.int32),
                    kb[b_idx, 3].astype(np.int32),
                    affinity,
                )
            )
            self.update_poses(
                a_id, b_id, all_keypoints, connections, pose_entries, pose_entry_size
            )
        pose_entries = np.asarray(pose_entries, dtype=np.float32).reshape(
            -1, pose_entry_size
        )
        return pose_entries[pose_entries[:, -1] >= 3], all_keypoints

    @staticmethod
    def convert_to_coco_format(pose_entries, all_keypoints):
        poses, scores = [], []
        for pose in pose_entries:
            keypoints = np.zeros(17 * 3, dtype=np.float32)
            for target_id, source_id in enumerate(COCO_FROM_OPENPOSE):
                point_id = int(pose[source_id])
                if point_id >= 0:
                    keypoints[target_id * 3 : target_id * 3 + 3] = all_keypoints[
                        point_id, :3
                    ]
            poses.append(keypoints)
            scores.append(pose[-2] * max(0, pose[-1] - 1))
        return np.asarray(poses), np.asarray(scores)


def _max_pool_nms(heatmaps):
    padded = np.pad(heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
    pooled = np.maximum.reduce(
        [
            padded[:, :, y : y + heatmaps.shape[2], x : x + heatmaps.shape[3]]
            for y in range(3)
            for x in range(3)
        ]
    )
    return heatmaps * (heatmaps == pooled)


def decode_poses(pafs, heatmaps, score_threshold=0.1):
    heatmaps, pafs = (
        np.asarray(heatmaps, dtype=np.float32),
        np.asarray(pafs, dtype=np.float32),
    )
    if heatmaps.shape != (1, 19, 32, 57) or pafs.shape != (1, 38, 32, 57):
        raise ValueError("expected heatmap [1,19,32,57] and PAF [1,38,32,57]")
    nms_heatmaps = _max_pool_nms(heatmaps)
    return OpenPoseDecoder(score_threshold=score_threshold)(
        heatmaps, nms_heatmaps, pafs
    )
