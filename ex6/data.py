
import numpy as np

raw_y = [607, 583, 521, 494, 369, 782, 570, 678, 467, 620, 425, 395, 346, 361,
     310, 300, 382, 294, 315, 323, 421, 339, 398, 328, 335, 291, 329, 310,
     294, 321, 286, 349, 279, 268, 293, 310, 259, 241, 243, 272, 247, 275,
     220, 245, 268, 357, 273, 301, 322, 276, 401, 368, 149, 507, 411, 362,
     358, 355, 362, 324, 332, 268, 259, 274, 248, 254, 242, 286, 276, 237,
     259, 251, 239, 247, 260, 237, 206, 242, 361, 267, 245, 331, 357, 284,
     263, 244, 317, 225, 254, 253, 251, 314, 239, 248, 250, 200, 256, 233,
     427, 391, 331, 395, 337, 392, 352, 381, 330, 368, 381, 316, 335, 316,
     302, 375, 361, 330, 351, 186, 221, 278, 244, 218, 126, 269, 238, 194,
     384, 154, 555, 387, 317, 365, 357, 390, 320, 316, 297, 354, 266, 279,
     327, 285, 258, 267, 226, 237, 264, 510, 490, 458, 425, 522, 927, 555,
     550, 516, 548, 560, 545, 633, 496, 498, 223, 222, 309, 244, 207, 258,
     255, 281, 258, 226, 257, 263, 266, 238, 249, 340, 247, 216, 241, 239,
     226, 273, 235, 251, 290, 473, 416, 451, 475, 406, 349, 401, 334, 446,
     401, 252, 266, 210, 228, 250, 265, 236, 289, 244, 327, 274, 223, 327,
     307, 338, 345, 381, 369, 445, 296, 303, 326, 321, 309, 307, 319, 288,
     299, 284, 278, 310, 282, 275, 372, 295, 306, 303, 285, 316, 294, 284,
     324, 264, 278, 369, 254, 306, 237, 439, 287, 285, 261, 299, 311, 265,
     292, 282, 271, 268, 270, 259, 269, 249, 261, 425, 291, 291, 441, 222,
     347, 244, 232, 272, 264, 190, 219, 317, 232, 256, 185, 210, 213, 202,
     226, 250, 238, 252, 233, 221, 220, 287, 267, 264, 273, 304, 294, 236,
     200, 219, 276, 287, 365, 438, 420, 396, 359, 405, 397, 383, 360, 387,
     429, 358, 459, 371, 368, 452, 358, 371]
raw_y = np.asarray(raw_y)

ind = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5, 5,
       5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7,
       7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10,
       10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11,
       11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12,
       12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13,
       13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
       15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18,
       18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
       20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21,
       21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23,
       23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
       24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
       25, 25, 25, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28,
       28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30,
       30, 30, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 33, 34, 34, 34, 34, 34,
       34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34]

is_child = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0]

is_child_row = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
                0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

# arg_is_child_row = 0, 1, 13, 14, 15, 16, 17, 22, 27

ind = np.asarray(ind, dtype=np.int) - 1
ids, cnts = np.unique(ind, return_counts=True)

is_child = np.asarray(is_child, dtype=np.bool)
is_child_row = np.asarray(is_child_row, dtype=np.bool)

x = np.arange(1, 21)

y = np.ma.empty(shape=(len(ids), max(cnts)))
mask = np.zeros_like(y, dtype=np.bool)

# Fill data
for i in ids:
    data = raw_y[ind==i]
    y[i, :] = np.concatenate([data, np.zeros(y.shape[-1] - len(data))])
    mask[i, len(data):] = 1
y.mask = mask


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    #
    # Note: Doing linear regression (with bias) on logarithmic data implies
    #       exponential decay to base level.
    #

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))

    for iRow, row in enumerate(y):
        color = 'salmon' if is_child_row[iRow] else 'gray'
        axs[0].plot(row, color=color, marker='o', linewidth=2.0, markersize=3, alpha=0.4)
        axs[1].plot(np.log(row), color=color, marker='o', linewidth=2.0, markersize=3, alpha=0.4)

    axs[0].plot(y[is_child_row, :].max(axis=0), linewidth=1.5, color='salmon', alpha=0.8)
    axs[0].plot(y[is_child_row, :].min(axis=0), linewidth=1.5, color='salmon', alpha=0.8)
    axs[0].plot(y[is_child_row, :].mean(axis=0), linewidth=1.5, color='red', alpha=0.8)
    axs[1].plot(np.log(y[is_child_row, :].max(axis=0)), linewidth=1.5, color='salmon', alpha=0.8)
    axs[1].plot(np.log(y[is_child_row, :].min(axis=0)), linewidth=1.5, color='salmon', alpha=0.8)
    axs[1].plot(np.log(y[is_child_row, :].mean(axis=0)), linewidth=1.5, color='red', alpha=0.8)

    axs[0].plot(y[np.logical_not(is_child_row), :].max(axis=0), linewidth=1.5, color='gray', alpha=0.8)
    axs[0].plot(y[np.logical_not(is_child_row), :].min(axis=0), linewidth=1.5, color='gray', alpha=0.8)
    axs[0].plot(y[np.logical_not(is_child_row), :].mean(axis=0), linewidth=1.5, color='black', alpha=0.8)
    axs[1].plot(np.log(y[np.logical_not(is_child_row), :].max(axis=0)), linewidth=1.5, color='gray', alpha=0.8)
    axs[1].plot(np.log(y[np.logical_not(is_child_row), :].min(axis=0)), linewidth=1.5, color='gray', alpha=0.8)
    axs[1].plot(np.log(y[np.logical_not(is_child_row), :].mean(axis=0)), linewidth=1.5, color='black', alpha=0.8)

    axs[0].set_ylabel('reaction time [ms]')
    axs[0].set_xlabel('attempt')
    axs[1].set_ylabel('reaction time [log(ms)]')
    axs[1].set_xlabel('attempt')

    axs[0].set_xlim([0, 19])
    axs[1].set_xlim([0, 19])

    axs[0].grid()
    axs[1].grid()

    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()

    plt.show()
