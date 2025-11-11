import numpy as np


def padding(array, values, axis):
    # This function should be doing post padding 0s.
    if axis not in {0, 1, 2}:
        print("Error! axis should be 0 or 1 or 2.")

    dim = array.shape
    new_dim = [0 for i in range(len(dim))]
    for i in range(len(dim)):
        if i == axis:
            new_dim[i] = dim[i] + 1
        else:
            new_dim[i] = dim[i]
    new_dim = tuple(new_dim)
    # print(new_dim)
    new_array = np.zeros(new_dim)

    if len(dim) == 2:
        for i in range(dim[0]):
            for j in range(dim[1]):
                new_array[i][j] = array[i][j]
    if len(dim) == 3:
        for i in range(dim[0]):
            for j in range(dim[1]):
                for k in range(dim[2]):
                    new_array[i][j][k] = array[i][j][k]
    return new_array


def adjust_dim(array):
    # Make the dim even
    # print(len(array.shape))
    for i in range(len(array.shape)):
        if array.shape[i] % 2 != 0:
            array = padding(array, 0, i)
    # if array.shape[1] % 2 != 0:
    #     array = padding(array, 0, 1)
    return array


def GAME_recursive(density, gt, currentLevel, targetLevel):
    if currentLevel == targetLevel:
        game = abs(np.sum(density) - np.sum(gt))
        return np.round(game, 3)

    else:
        if len(density.shape) == 2:
            # print("2D")
            density = adjust_dim(density)
            gt = adjust_dim(gt)
            density_slice = [];
            gt_slice = []

            density_slice.append(density[0:density.shape[0] // 2, 0:density.shape[1] // 2])
            density_slice.append(density[0:density.shape[0] // 2, density.shape[1] // 2:])
            density_slice.append(density[density.shape[0] // 2:, 0:density.shape[1] // 2])
            density_slice.append(density[density.shape[0] // 2:, density.shape[1] // 2:])
            # for i in range(4):
            #     print(np.sum(density_slice[i]))
            # print(np.sum(density_slice))

            gt_slice.append(gt[0:gt.shape[0] // 2, 0:gt.shape[1] // 2])
            gt_slice.append(gt[0:gt.shape[0] // 2, gt.shape[1] // 2:])
            gt_slice.append(gt[gt.shape[0] // 2:, 0:gt.shape[1] // 2])
            gt_slice.append(gt[gt.shape[0] // 2:, gt.shape[1] // 2:])
            # for i in range(4):
            #     print(np.sum(gt_slice[i]))
            # print(np.sum(gt_slice))
            currentLevel = currentLevel + 1;
            res = []
            for a in range(4):
                res.append(GAME_recursive(density_slice[a], gt_slice[a], currentLevel, targetLevel))
            game = sum(res)
            return np.round(game, 3)

        if len(density.shape) == 3:
            # print("3D")
            density = adjust_dim(density)
            gt = adjust_dim(gt)
            density_slice = [];
            gt_slice = []

            density_slice.append(density[0:density.shape[0] // 2, 0:density.shape[1] // 2, 0:density.shape[2] // 2])
            density_slice.append(density[0:density.shape[0] // 2, 0:density.shape[1] // 2, density.shape[2] // 2:])
            density_slice.append(density[0:density.shape[0] // 2, density.shape[1] // 2:, 0:density.shape[2] // 2])
            density_slice.append(density[0:density.shape[0] // 2, density.shape[1] // 2:, density.shape[2] // 2:])
            density_slice.append(density[density.shape[0] // 2:, 0:density.shape[1] // 2, 0:density.shape[2] // 2])
            density_slice.append(density[density.shape[0] // 2:, 0:density.shape[1] // 2, density.shape[2] // 2:])
            density_slice.append(density[density.shape[0] // 2:, density.shape[1] // 2:, 0:density.shape[2] // 2])
            density_slice.append(density[density.shape[0] // 2:, density.shape[1] // 2:, density.shape[2] // 2:])

            # for i in range(4):
            #     print(np.sum(density_slice[i]))
            # print(np.sum(density_slice))
            gt_slice.append(gt[0:gt.shape[0] // 2, 0:gt.shape[1] // 2, 0:gt.shape[2] // 2])
            gt_slice.append(gt[0:gt.shape[0] // 2, 0:gt.shape[1] // 2, gt.shape[2] // 2:])
            gt_slice.append(gt[0:gt.shape[0] // 2, gt.shape[1] // 2:, 0:gt.shape[2] // 2])
            gt_slice.append(gt[0:gt.shape[0] // 2, gt.shape[1] // 2:, gt.shape[2] // 2:])
            gt_slice.append(gt[gt.shape[0] // 2:, 0:gt.shape[1] // 2, 0:gt.shape[2] // 2])
            gt_slice.append(gt[gt.shape[0] // 2:, 0:gt.shape[1] // 2, gt.shape[2] // 2:])
            gt_slice.append(gt[gt.shape[0] // 2:, gt.shape[1] // 2:, 0:gt.shape[2] // 2])
            gt_slice.append(gt[gt.shape[0] // 2:, gt.shape[1] // 2:, gt.shape[2] // 2:])

            # for i in range(4):
            #     print(np.sum(gt_slice[i]))
            # print(np.sum(gt_slice))
            currentLevel = currentLevel + 1;
            res = []
            for a in range(8):
                res.append(GAME_recursive(density_slice[a], gt_slice[a], currentLevel, targetLevel))
            game = sum(res)
            return np.round(game, 3)

def GAME_metric(preds, gts, l):
    # res = []
    # for i in range(len(gts)):
    #     res.append(GAME_recursive(preds[i], gts[i], 0, l))
    res = GAME_recursive(preds, gts, 0, l)
    return res#np.mean(res)

if __name__ == "__main__":
    a = np.random.randint(0,2,(1,192,159,20))
    print(np.sum(a))
    # print(a.shape)
    b = np.random.randint(0,2,(1,192,159,20))
    print(np.sum(b))
    res = GAME_metric(a,b,1)
    print(res)