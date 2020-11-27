import cv2
import numpy as np
import imutils
from typing import Tuple


def mse(image1: np.ndarray, image2: np.ndarray) -> float:
    err = float(np.sum((image1.astype("float") - image2.astype("float")) ** 2))
    err /= float(image1.shape[0] * image1.shape[1])

    return err


def measure_similarity(aligned: np.ndarray, template: np.ndarray) -> float:
    result = cv2.matchTemplate(cv2.Canny(imutils.resize(template, width=600), 30, 100),
                               cv2.Canny(imutils.resize(aligned, width=600), 30, 100),
                               cv2.TM_CCORR_NORMED)
    (_, max_val, _, _) = cv2.minMaxLoc(result)

    return max_val


def check_right_filling(aligned: np.ndarray, template: np.ndarray, fields: list, threshold: float = 5000,
                        debug_mode: int = 0) -> Tuple[float, np.ndarray]:
    debug_aligned: np.ndarray = aligned.copy()
    debug_aligned = cv2.cvtColor(debug_aligned, cv2.COLOR_GRAY2BGR)

    num_of_filled = 0
    for field in fields:
        y, x = field[1], field[2]
        img1 = cv2.Canny(aligned[y[1]:y[0], x[1]:x[0]], 3, 100)
        img2 = cv2.Canny(template[y[1]:y[0], x[1]:x[0]], 3, 100)
        mse_score = mse(img1, img2)

        if mse_score > threshold:
            num_of_filled += 1

        if debug_mode == 1:
            color = ((0, 47, 31) if mse_score > threshold else (0, 0, 255))
            cv2.rectangle(debug_aligned, (x[0], y[0]), (x[1], y[1]), color, 3)
            cv2.putText(debug_aligned, str(int(mse_score)), (x[1], y[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    percent_filled = num_of_filled / len(fields)
    return percent_filled, debug_aligned
