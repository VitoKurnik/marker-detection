import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv2.imread(str(sys.argv[1]), 0)  # marker
#img2 = cv2.imread(str(sys.argv[2]), 0)

camera = cv2.VideoCapture(0)
return_value, image = camera.read()
cv2.imwrite('capture.jpg', image)
img2 = cv2.imread('capture.jpg', 0)

orb = cv2.ORB_create()

# poiščemo značilnice
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)

# Lowe's ratio test - https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
good = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # poiščemo transformacijo markerja
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    h, w = img1.shape  # velikost markerja
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)  # poiščemo objekt

    mean_x1 = 0.5 * (dst[0][0][0] + dst[1][0][0])
    mean_x2 = 0.5 * (dst[2][0][0] + dst[3][0][0])
    mean_y1 = 0.5 * (dst[0][0][1] + dst[3][0][1])
    mean_y2 = 0.5 * (dst[1][0][1] + dst[2][0][1])
    position_x = (mean_x1 + mean_x2) / 2
    position_y = (mean_y1 + mean_y2) / 2

    img2 = cv2.polylines(img2, [np.int32(dst)], True, (255, 0, 0, 0), 3, cv2.LINE_AA)  # narišemo pravoktonik okoli markerja
    img2 = cv2.circle(img2, (np.int(position_x), np.int(position_y)), 10, 0, 10, 6, 0)  # narišemo krog v sredini markerja
    img2 = cv2.putText(img2, str("X:%d Y:%d" % (position_x, position_y)), (np.int(position_x + 20), np.int(position_y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 6)  # izpišemo koordinate markerja

    plt.imshow(img2, 'gray'), plt.show()
else:
    print("Izbran marker ni bil najden na zajeti sliki")
    sys.exit()
