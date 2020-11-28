import cv2
import numpy as np


def match_images(matches, keep_percent, image_orb_features, template_orb_features):
    (kps1, descs1) = image_orb_features
    (kps2, descs2) = template_orb_features

    if not matches:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
        matches = matcher.match(descs1, descs2)
        matches = sorted(matches, key=lambda x: x.distance)

    keep = int(len(matches) * keep_percent)
    matches = matches[:keep]

    pts1 = np.zeros((len(matches), 2), dtype="float")
    pts2 = np.zeros((len(matches), 2), dtype="float")

    for (i, m) in enumerate(matches):
        pts1[i] = kps1[m.queryIdx].pt
        pts2[i] = kps2[m.trainIdx].pt

    return pts1, pts2


def find_homography(image_to_align, template, pts1, pts2):
    (H, mask) = cv2.findHomography(pts1, pts2, method=cv2.RANSAC)

    (h, w) = template.shape[:2]
    aligned_image = cv2.warpPerspective(image_to_align, H, (w, h))

    return aligned_image

