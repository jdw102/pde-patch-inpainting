import cv2 #pip3 install opencv-python
import numpy as np
from scipy.ndimage import gaussian_filter #pip3 install scipy

a = cv2.imread('../inputs/transfer/rice.jpg').astype(float) / 255.0  # Input Texture
b = cv2.imread('../inputs/transfer/bill.png').astype(float) / 255.0  # Input Image
cv2.imshow('a', a)
cv2.imshow('b', b)
cv2.waitKey(0)
cv2.destroyAllWindows()

if a.ndim == 2:
    a = np.repeat(a[:, :, np.newaxis], 3, axis=2)
if b.ndim == 2:
    b = np.repeat(b[:, :, np.newaxis], 3, axis=2)

inputTexture = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
inputTarget = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

mask_in = inputTexture < -1
mask_out = inputTarget < -1
inputTexture[mask_in] = -1
inputTarget[mask_out] = -1

m, n = inputTarget.shape
w = 8
al = 0.43
o = round(w / 3)
m1 = (m - o) // w * w + o
n1 = (n - o) // w * w + o
outputTexture = np.zeros((m, n))
outputTexture1 = np.zeros((m, n, 3))
iter = 2

for p in range(iter):
    for i in range(1, m1 // w + 1):
        for j in range(1, n1 // w + 1):
            if np.all(mask_out[(i - 1) * w:i * w + o, (j - 1) * w:j * w + o]):
                outputTexture[(i - 1) * w:i * w + o, (j - 1) * w:j * w + o] = 0
                outputTexture1[(i - 1) * w:i * w + o, (j - 1) * w:j * w + o, :] = 0
                continue
            mask = np.zeros((w + o, w + o))
            temp1 = outputTexture[(i - 1) * w:i * w + o, (j - 1) * w:j * w + o]
            if i == 1 and j == 1:
                nearPatch, nearPatch1 = givePatch1(al, a, inputTexture[0:w + o, 0:w + o], inputTarget[0:w + o, 0:w + o], temp1, mask)
                outputTexture[0:w + o, 0:w + o] = nearPatch
                outputTexture1[0:w + o, 0:w + o, :] = nearPatch1
                continue
            elif i == 1:
                mask[:, 0:o] = 1
                nearPatch, nearPatch1 = g

for p in range(1, iter+1):
    for i in range(1, math.floor(m1/w)+1) + [(m-o)/w]:
        for j in range(1, math.floor(n1/w)+1) + [(n-o)/w]:
            if j == 1:
                mask[0:o, :] = 1
                nearPatch, nearPatch1 = givePatch1(al, a, inputTexture, inputTarget[(i-1)*w:i*w+o, (j-1)*w:j*w+o], temp1, mask)
                error = (nearPatch*mask - temp1*mask)**2
                error = error[0:o, :]
                cost, path = findBoundaryHelper1(error.T)
                boundary = np.zeros((w+o, w+o))
                ind = np.argmin(cost[0, :])
                boundary[0:o, :] = findBoundaryHelper2(path, ind).T
            else:
                mask[:, 0:o] = 1
                mask[0:o, :] = 1
                nearPatch, nearPatch1 = givePatch1(al, a, inputTexture, inputTarget[(i-1)*w:i*w+o, (j-1)*w:j*w+o], temp1, mask)
                error = (nearPatch*mask - temp1*mask)**2
                error1 = error[0:o, :]
                cost1, path1 = findBoundaryHelper1(error1.T)
                error2 = error[:, 0:o]
                cost2, path2 = findBoundaryHelper1(error2)
                cost = cost1[0:o, :] + cost2[0:o, :]
                boundary = np.zeros((w+o, w+o))
                ind = np.argmin(np.diag(cost))
                boundary[0:o, ind:w+o] = findBoundaryHelper2(path1[ind:o+w, :], o-ind+1).T
                boundary[ind:o+w, 0:o] = findBoundaryHelper2(path2[ind:o+w, :], ind).T
                boundary[0:ind-1, 0:ind-1] = 1


            smoothBoundary = gaussian_filter(boundary, sigma=1.5)
            smoothBoundary1 = np.tile(boundary, (1, 1, 3))
            temp2 = temp1 * smoothBoundary + nearPatch * (1 - smoothBoundary)
            outputTexture[(i-1)*w+1:i*w+o, (j-1)*w+1:j*w+o] = temp2
            outputTexture1[(i-1)*w+1:i*w+o, (j-1)*w+1:j*w+o, :] = outputTexture1[(i-1)*w+1:i*w+o, (j-1)*w+1:j*w+o, :] * smoothBoundary1 + nearPatch1 * (1 - smoothBoundary1)





output = outputTexture1[0:m, 0:n, :]
output[np.tile(mask_out, (1, 1, 3))] = 0
w = round(w * 0.7)
o = round(w / 3)

if iter > 1:
    al = 0.8 * (p - 1) / (iter - 1) + 0.1
else:
    pass

inputTarget = outputTexture
inputTexture[mask_in] = -1
inputTarget[mask_out] = -1
m, n = inputTarget.shape
m1 = (m - o) // w * w + o
n1 = (n - o) // w * w + o
outputTexture = np.zeros((m, n))
outputTexture1 = np.zeros((m, n, 3))
plt.figure()
plt.imshow(output)
plt.show()

