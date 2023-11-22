import numpy as np
from scipy.ndimage import gaussian_filter
from src.translation_attempt.findBoundryHelper1 import find_boundary_helper1
from src.translation_attempt.findBoundryHelper2 import find_boundary_helper2
from src.translation_attempt.givePatch1 import give_patch1
import cv2 as cv


def texture_transfer(a, b):

    # a = cv2.imread(input_texture).astype(float) / 255.0  # Input Texture
    # b = cv2.imread(input_image).astype(float) / 255.0  # Input Image

    # cv.imshow('Input Texture', a)
    # cv.imshow('Input Image', b)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    if a.ndim != 3:
        a = np.tile(a, (1, 1, 3))
    if b.ndim != 3:
        b = np.tile(b, (1, 1, 3))

    a_8bit = (a * 255).astype(np.uint8)
    b_8bit = (b * 255).astype(np.uint8)

    # Convert to grayscale
    inputTexture = cv.cvtColor(a_8bit, cv.COLOR_BGR2GRAY) /  255.0
    inputTarget = cv.cvtColor(b_8bit, cv.COLOR_BGR2GRAY) / 255.0

    mask_in = inputTexture < -1
    mask_out = inputTarget < -1

    inputTexture[mask_in] = -1
    inputTarget[mask_out] = -1

    m, n = inputTarget.shape
    w = 4
    al = 0.43
    o = round(w / 3)
    m1 = (m - o) // w * w + o
    n1 = (n - o) // w * w + o

    outputTexture = np.zeros((m, n))
    outputTexture1 = np.zeros((m, n, 3))

    iterator = 2
    i_arr = [i for i in range(m1 // w)]
    # i_arr.append((m - o) // w)
    j_arr = [j for j in range(n1 // w)]
    # j_arr.append((n - o) // w)

    for p in range(iterator):
        for i in i_arr:
            for j in j_arr:
                z = i * w
                y = (i + 1) * w + o
                x = j * w
                zz = (j + 1) * w + o
                t = mask_out[i * w:(i + 1) * w + o, j * w:(j + 1) * w + o]
                if np.all(np.all(t)):
                    outputTexture[i * w:(i + 1) * w + o, j * w:(j + 1) * w + o] = 0
                    outputTexture1[i * w:(i + 1) * w + o, j * w:(j + 1) * w + o, :] = 0
                    continue
                mask = np.zeros((w + o, w + o))
                temp1 = outputTexture[i * w:(i + 1) * w + o, j * w:(j + 1) * w + o]
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
                                                        inputTarget[i * w:(i + 1) * w + o, j * w:(j + 1) * w + o],
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
                                                        inputTarget[i * w:(i + 1) * w + o, j * w:(j + 1) * w + o],
                                                        temp1,
                                                        mask)
                    error = (nearPatch * mask - temp1 * mask) ** 2
                    error = error[0:o, :]
                    cost, path = find_boundary_helper1(error.T)
                    boundary = np.zeros((w + o, w + o))
                    _, ind = np.unravel_index(np.argmin(cost[0, :]), cost.shape)
                    boundary[0:o, :] = find_boundary_helper2(path, ind).T
                else:
                    mask[:, :o] = 1
                    mask[:o, :] = 1

                    nearPatch, nearPatch1 = give_patch1(al, a, inputTexture,
                                                       inputTarget[i * w:(i + 1) * w + o, j * w:(j + 1) * w + o], temp1,
                                                       mask)

                    error = (nearPatch * mask - temp1 * mask) ** 2
                    error1 = error[:o, :]
                    cost1, path1 = find_boundary_helper1(error1.T)

                    error2 = error[:, :o]
                    cost2, path2 = find_boundary_helper1(error2)

                    cost = cost1[:o, :] + cost2[:o, :]
                    boundary = np.zeros((w + o, w + o))
                    _, ind = np.unravel_index(np.argmin(np.diag(cost)), cost.shape)
                    boundary[:o, ind:w + o] = find_boundary_helper2(path1[ind:o + w, :], o - ind).T

                    boundary[ind:o + w, :o] = find_boundary_helper2(path2[ind:o + w, :], ind - 1)

                    boundary[:ind - 1, :ind - 1] = 1

                smoothBoundary = gaussian_filter(boundary, sigma=1.5)
                smoothBoundary1 = np.tile(boundary[:, :, np.newaxis], (1, 1, 3))
                temp2 = temp1 * smoothBoundary + nearPatch * (1 - smoothBoundary)
                outputTexture[i * w:(i + 1) * w + o, j * w:(j + 1) * w + o] = temp2
                outputTexture1[i * w:(i + 1) * w + o, j * w:(j + 1) * w + o, :] = outputTexture1[
                                                                                          i * w:(i + 1) * w + o, j * w:(j + 1) * w + o,
                                                                                          :] * smoothBoundary1 + nearPatch1 * (
                                                                                                  1 - smoothBoundary1)

        output = outputTexture1[0:m, 0:n, :]
        # output[np.tile(mask_out[:, :, np.newaxis], (1, 1, 3))] = 0
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
        cv.imshow("iter " + str(p), output)
        cv.waitKey(0)

# if __name__ == "__main__":
#     texture_image = cv.imread("../../data/texture.jpg")
#     image = cv.imread("../../data/restored.jpg")
#     texture_transfer(texture_image.astype(float) / 255.0, image.astype(float) / 255.0)
