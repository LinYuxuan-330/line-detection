import cv2
import numpy as np

# 读取图像
img = cv2.imread('image05.png')

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 创建LSD检测器对象
lsd = cv2.createLineSegmentDetector()

# 检测直线
lines, widths, _, _ = lsd.detect(gray)

# 在原图上画出直线
for i in range(len(lines)):
    x1 = int(round(lines[i][0][0]))
    y1 = int(round(lines[i][0][1]))
    x2 = int(round(lines[i][0][2]))
    y2 = int(round(lines[i][0][3]))
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 显示图像
# cv2.namedWindow('LSD Line Detection',cv2.WINDOW_NORMAL)
cv2.imshow('LSD Line Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()