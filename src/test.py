import cv2
import numpy as np

file = r'D:\Stellarium\screenshots\dataset\image\0.npz'
data = np.load(file)

print(data['data'])

cv2.imshow('xd', data['data'])
cv2.waitKey(0)