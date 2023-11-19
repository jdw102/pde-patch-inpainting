import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from src.findBoundryHelper1 import find_boundary_helper1
from src.findBoundryHelper2 import find_boundary_helper2
from src.givePatch1 import give_patch1
import cv2 as cv
import matplotlib.pyplot as plt
from image_util import extract_rectangle

def texture_transfer(input_texture, input_image):
    print("transfering texture...")
    a = (input_texture.astype(np.float32) / 255.0).clip(0.0, 1.0)  # Input Texture
    b = (input_image.astype(np.float32) / 255.0).clip(0.0, 1.0)
    # cv2.imshow('a', a)
    # cv2.imshow('b', b)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if a.ndim == 2:
        a = np.repeat(a[:, :, np.newaxis], 3, axis=2)
    if b.ndim == 2:
        b = np.repeat(b[:, :, np.newaxis], 3, axis=2)

    inputTexture = cv.cvtColor(a, cv.COLOR_BGR2GRAY)
    inputTarget = cv.cvtColor(b, cv.COLOR_BGR2GRAY)

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
    iterator = 2

    for p in range(iterator):
        for i in range(m1 // w + 1):
            for j in range(n1 // w + 1):
                print(i, j)
                if np.all(mask_out[(i - 1) * w:i * w + o, (j - 1) * w:j * w + o]):
                    outputTexture[(i - 1) * w:i * w + o, (j - 1) * w:j * w + o] = 0
                    outputTexture1[(i - 1) * w:i * w + o, (j - 1) * w:j * w + o, :] = 0
                    continue
                mask = np.zeros((w + o, w + o))
                temp1 = outputTexture[(i - 1) * w:i * w + o, (j - 1) * w:j * w + o]
                if i == 0 and j == 0:
                    nearPatch, nearPatch1 = give_patch1(al, a, inputTexture[0:w + o, 0:w + o],
                                                        inputTarget[0:w + o, 0:w + o],
                                                        temp1, mask)
                    outputTexture[0:w + o, 0:w + o] = nearPatch
                    outputTexture1[0:w + o, 0:w + o, :] = nearPatch1
                    continue
                elif i == 0:
                    mask[:, 0:o] = 1
                    nearPatch, nearPatch1 = give_patch1(al, a, inputTexture,
                                                        inputTarget[(i - 1) * w:(i * w) + o, (j - 1) * w:(j * w) + o],
                                                        temp1,
                                                        mask)
                    error = (nearPatch * mask - temp1 * mask) ** 2
                    error = error[:, 0:o]
                    cost, path = find_boundary_helper1(error)
                    boundary = np.zeros((w + o, w + o))
                    _, ind = np.unravel_index(np.argmin(cost[0, :]), cost.shape)
                    boundary[:, 0:o] = find_boundary_helper2(path, ind)
                elif j == 0:
                    mask[0:o, :] = 1
                    nearPatch, nearPatch1 = give_patch1(al, a, inputTexture,
                                                        inputTarget[(i - 1) * w:(i * w) + o, (j - 1) * w:(j * w) + o],
                                                        temp1,
                                                        mask)
                    error = (nearPatch * mask - temp1 * mask) ** 2
                    error = error[0:o, :]
                    cost, path = find_boundary_helper1(error.T)
                    boundary = np.zeros((w + o, w + o))
                    _, ind = np.unravel_index(np.argmin(cost[0, :]), cost.shape)
                    boundary[0:o, :] = find_boundary_helper2(path, ind).T
                else:
                    mask[:, 0:o] = 1
                    mask[0:o, :] = 1
                    nearPatch, nearPatch1 = give_patch1(al, a, inputTexture,
                                                        inputTarget[(i - 1) * w:(i * w) + o, (j - 1) * w:(j * w) + o],
                                                        temp1,
                                                        mask)
                    error = (nearPatch * mask - temp1 * mask) ** 2
                    error1 = error[0:o, :]
                    cost1, path1 = find_boundary_helper1(error1.T)
                    error2 = error[:, 0:o]
                    cost2, path2 = find_boundary_helper1(error2)

                    cost = cost1[0:o, :] + cost2[0:o, :]
                    boundary = np.zeros((w + o, w + o))
                    _, ind = np.unravel_index(np.argmin(np.diag(cost)), cost.shape)
                    boundary[0:o, ind:w + o] = find_boundary_helper2(path1[ind:o + w, :], o - ind + 1).T
                    boundary[ind:o + w, 0:o] = find_boundary_helper2(path2[ind:o + w, :], ind).T
                    boundary[0:ind - 1, 0:ind - 1] = 1

                smoothBoundary = gaussian_filter(boundary, sigma=1.5)
                smoothBoundary1 = np.tile(boundary, (1, 1, 3))
                temp2 = temp1 * smoothBoundary + nearPatch * (1 - smoothBoundary)
                outputTexture[(i - 1) * w + 1:i * w + o, (j - 1) * w + 1:j * w + o] = temp2
                outputTexture1[(i - 1) * w + 1:i * w + o, (j - 1) * w + 1:j * w + o, :] = outputTexture1[
                                                                                          (i - 1) * w + 1:i * w + o,
                                                                                          (j - 1) * w + 1:j * w + o,
                                                                                          :] * smoothBoundary1 + nearPatch1 * (
                                                                                                  1 - smoothBoundary1)

    output = outputTexture1[0:m, 0:n, :]
    output[np.tile(mask_out, (1, 1, 3))] = 0
    w = round(w * 0.7)
    o = round(w / 3)

    if iterator > 1:
        al = 0.8 * (p - 1) / (iterator - 1) + 0.1
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

if __name__ == "__main__":
    texture_image = cv.imread("../data/Granite-Rock.jpg")
    texture = extract_rectangle(texture_image, (100, 50, 100, 100))
    image = cv.imread("../data/cat_drawing.jpg")
    texture_transfer(texture, image)