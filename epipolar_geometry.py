import cv2
import numpy as np

# Load stereo images
img1 = cv2.imread("D:/Downlaods/Taj_Mahal_(Edited).jpeg")  # First image
img2 = cv2.imread("D:/Downlaods/taj_mahal_at_morning_from_south-east.webp")  # Second image

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect SIFT keypoints and descriptors
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# FLANN-based matcher parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test
good_matches = []
pts1 = []
pts2 = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

pts1 = np.float32(pts1)
pts2 = np.float32(pts2)

# Compute the Fundamental Matrix using RANSAC
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

# Select inlier points
inliers1 = pts1[mask.ravel() == 1]
inliers2 = pts2[mask.ravel() == 1]

# Function to draw epipolar lines
def draw_epipolar_line(img, F, point, img_shape):
    # Convert point to homogeneous coordinates
    point_h = np.array([point[0], point[1], 1]).reshape(3, 1)
    
    # Epipolar line: l = F * x
    line = F @ point_h
    
    # Line equation: ax + by + c = 0
    a, b, c = line.flatten()
    
    # Compute intersection points with image borders
    x0, y0 = 0, int(-c / b) if b != 0 else 0
    x1, y1 = img_shape[1], int(-(a * img_shape[1] + c) / b) if b != 0 else img_shape[0]
    
    # Draw the line
    img_with_line = img.copy()
    cv2.line(img_with_line, (x0, y0), (x1, y1), (0, 255, 0), 2)
    return img_with_line

# Mouse callback function to select a point
def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        img_with_point = img1.copy()
        cv2.circle(img_with_point, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Image 1', img_with_point)

        img2_with_line = draw_epipolar_line(img2, F, (x, y), img2.shape)
        cv2.imshow('Image 2', img2_with_line)

# Display images
cv2.imshow('Image 1', img1)
cv2.imshow('Image 2', img2)
cv2.setMouseCallback('Image 1', select_point)

cv2.waitKey(0)
cv2.destroyAllWindows()
