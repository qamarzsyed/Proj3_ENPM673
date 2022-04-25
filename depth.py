import numpy as np
import cv2
import scipy

# dataset 1 = curule, 2 = octagon, 3 = pendulum
dataset_num = 1

# selection between 3 diff datasets
if dataset_num == 1:
    K = [[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]]
    Kt = np.transpose(K)
    f = K[0][0]
    baseline = 88.39
    w = 1920
    h = 1080
    ndisp = 220
    image0 = cv2.imread('data/curule/im0.png')
    image1 = cv2.imread('data/curule/im1.png')
    dataset = 'curule'

if dataset_num == 2:
    K = [[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]]
    Kt = np.transpose(K)
    f = K[0][0]
    baseline = 221.76
    w = 1920
    h = 1080
    ndisp = 100
    image0 = cv2.imread('data/octagon/im0.png')
    image1 = cv2.imread('data/octagon/im1.png')
    dataset = 'octagon'

if dataset_num == 3:
    K = [[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]]
    Kt = np.transpose(K)
    f = K[0][0]
    baseline = 221.76
    w = 1920
    h = 1080
    ndisp = 180
    image0 = cv2.imread('data/pendulum/im0.png')
    image1 = cv2.imread('data/pendulum/im1.png')
    dataset = 'pendulum'

# image to grayscale
g0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
g1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# detect features using BRISK
# use this as it doesnt need the contrib library and is not licensed
detect = cv2.BRISK_create()

kp_0, coord_0 = detect.detectAndCompute(image0, None)
kp_1, coord_1 = detect.detectAndCompute(image1, None)


# Brute force matcher to match features to each other from different perspectives
bfm = cv2.BFMatcher_create(cv2.NORM_HAMMING, True)
matches = bfm.match(coord_0, coord_1)

# select top matches for an n number
n = 30
if n > len(matches):
    n = len(matches)
top_matches = sorted(matches, key=lambda x: x.distance)[:n]

points_0 = np.array([kp_0[match.queryIdx].pt for match in top_matches])
points_1 = np.array([kp_1[match.trainIdx].pt for match in top_matches])

# create the A matrix for the 8 pt algorithm
A = []
for i in range(n):
    x = points_0[i][0]
    y = points_0[i][1]
    xn = points_1[i][0]
    yn = points_1[i][1]

    A.append([x*xn, x*yn, x, y*xn, y*yn, y, xn, yn, 1])
A = np.array(A)

# solve for 8 pt algorithm to get F
U, S, Vt = np.linalg.svd(A)
V = np.transpose(Vt)
F = V[:, -1]

# reshape F from least squares and SVD solution, alternative RANSAC solution also commented out
F = np.reshape(F, (3,3))
# F, inliers = cv2.findFundamentalMat(points_0, points_1, method=cv2.FM_RANSAC)
# points_0 = points_0[inliers.ravel() == 1]
# points_1 = points_1[inliers.ravel() == 1]
points_0 = np.float32(points_0)
points_1 = np.float32(points_1)

# getting Essential matrix from F matrix and Camera matrix and solving for R and T
E = np.dot(np.dot(Kt, F), K)
U, S, Vt = np.linalg.svd(E)
S = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
E = np.dot(np.dot(U, S), Vt)

U, S, Vt = np.linalg.svd(E)
W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])


# Getting R and T from E
R, T = [], []
R.append(np.dot(U, np.dot(W, Vt)))
R.append(np.dot(U, np.dot(W, Vt)))
R.append(np.dot(U, np.dot(W.T, Vt)))
R.append(np.dot(U, np.dot(W.T, Vt)))
T.append(U[:, 2])
T.append(-U[:, 2])
T.append(U[:, 2])
T.append(-U[:, 2])
# positive depth check
for i in range(4):
    if (np.linalg.det(R[i]) < 0):
        R[i] = -R[i]
        T[i] = -T[i]
print(R)
print(T)


# Rectifying images
# allowed to use inbuilt functions for this portion so did
# found homography matrixes to rectify both images and then applied them to images
ret, H0, H1 = cv2.stereoRectifyUncalibrated(points_0, points_1, F, imgSize=(w, h))
print(H0)
print(H1)
rect_0 = cv2.warpPerspective(g0, H0, (w, h))
rect_1 = cv2.warpPerspective(g1, H1, (w, h))

# applied homography matrix transform to matching points as well
print(points_0.shape)
rect_pts0 = cv2.perspectiveTransform(points_0.reshape(-1, 1, 2), H0).reshape(-1, 2)
rect_pts1 = cv2.perspectiveTransform(points_1.reshape(-1, 1, 2), H1).reshape(-1, 2)
print(rect_pts0)

# Applied homography transform to fundamental matrix as well to find new fundamental matrix in rectified space
Fr = np.dot(np.linalg.inv(H1.T), np.dot(F, np.linalg.inv(H0)))
print(Fr)

# find epipolar lines between both rectified images
lines = cv2.computeCorrespondEpilines(rect_pts1, 2, Fr.T)
print(lines)
rect = np.concatenate((rect_0, rect_1), axis=1)

# drew the epipolar lines on the image with both rectified images
# also stored the line heights for these
# a is generally approximately 0 and b is approximately -1 so the epipolar lines were just horizontal lines at c height
line_heights = []
for line in lines:
    line = np.ravel(line)
    a, b, c = line
    line_heights.append(c)
    x1 = 0
    x2 = w*2
    y1 = c
    y2 = c
    cv2.line(rect, (x1, y1), (x2, y2), (0, 0, 255))

# SSD process to find disparity between the two rectified images
kernel = 5
disp_num = 15
ssd_img = np.zeros((h, w))
for y in range(kernel, h-kernel-1):
    for x in range(kernel + disp_num, w-kernel-1):
        ssd = np.empty([disp_num, 1])
        l = rect_0[(y - kernel):(y + kernel), (x - kernel):(x + kernel)]
        height, width = l.shape
        for d in range(0, disp_num):
            r = rect_1[(y - kernel):(y + kernel), (x - d - kernel):(x - d + kernel)]
            ssd[d] = np.sum((l[:, :] - r[:, :]) ** 2)
        ssd_img[y, x] = np.argmin(ssd)
ssd_img = ((ssd_img/ssd_img.max())*255).astype(np.uint8)


# get depth from triangulation
result = np.zeros(shape=ssd_img.shape).astype(float)
result[ssd_img > 0] = (f * baseline) / (ssd_img[ssd_img > 0])

depth = ((result/result.max())*255).astype(np.uint8)

# color map the greyscale
disp_cs = cv2.applyColorMap(ssd_img, cv2.COLORMAP_JET)
depth_cs = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

# save all images
cv2.imwrite(dataset + '_disp.png', ssd_img)
cv2.imwrite(dataset + '_disp_cs.png', disp_cs)

cv2.imwrite(dataset+ '_depth.png', depth)
cv2.imwrite(dataset+ '_depth_cs.png', depth_cs)

cv2.imwrite(dataset+'_rect.png', rect)




