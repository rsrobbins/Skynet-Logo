import cv2
import numpy as np

img = cv2.imread("Skynet-Terminator-Logo-1000x563.jpg")

# Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Threshold red
lower1 = np.array([0, 100, 100])
upper1 = np.array([10, 255, 255])

lower2 = np.array([170, 100, 100])
upper2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower1, upper1)
mask2 = cv2.inRange(hsv, lower2, upper2)

mask = cv2.bitwise_or(mask1, mask2)

# Keep only upper portion of image
mask[260:, :] = 0

# Find contours
contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Find contour nearest image center
h, w = mask.shape
center_x = w / 2

best = None
best_score = float("inf")

for cnt in contours:
    area = cv2.contourArea(cnt)

    if area < 5000:
        continue

    M = cv2.moments(cnt)

    if M["m00"] == 0:
        continue

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    score = abs(cx - center_x)

    if score < best_score:
        best_score = score
        best = cnt

# Fit rotated rectangle
rect = cv2.minAreaRect(best)

(center_x, center_y), (width, height), angle = rect

print("Center:", center_x, center_y)
print("Width :", width)
print("Height:", height)
print("Angle :", angle)

# Corner points
box = cv2.boxPoints(rect)
box = np.int32(box)

print("\nCorners:")
for p in box:
    print(tuple(p))

# Draw result
out = img.copy()
cv2.drawContours(out, [box], 0, (0,255,0), 2)

cv2.imwrite("detected_square.png", out)