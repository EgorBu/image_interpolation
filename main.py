from argparse import ArgumentParser

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def random_rect(max_x, max_y, max_w, max_h):
    x_r = np.random.randint(0, max_x)
    y_r = np.random.randint(0, max_y)
    width = np.random.randint(0, min(max_x - x_r, max_w))
    height = np.random.randint(0, min(max_y - y_r, max_h))
    return x_r, y_r, width, height


def squared_error(a, b):
    assert a.shape == b.shape
    return np.sum((a - b) ** 2)


def mean_squared_error(a, b):
    assert a.shape == b.shape
    return squared_error(a, b) / (a.shape[0] * a.shape[1])


def generate_population(size, max_x, max_y, max_w, max_h):
    for _ in range(size):
        yield random_rect(max_x, max_y, max_w, max_h)


def suggest_next(image, gen_image, max_x, max_y, max_w, max_h, population_size):
    res = None
    min_err = np.inf

    for sample in generate_population(size=population_size, max_x=max_x, max_y=max_y,
                                      max_w=max_w, max_h=max_h):
        x_0, y_0, width, height = sample
        x_1, y_1 = x_0 + width, y_0 + height

        im_slice = image[x_0:x_1, y_0:y_1]
        im_avg = np.mean(im_slice, axis=(0, 1))

        gen_slice = gen_image[x_0:x_1, y_0:y_1]
        gen_avg = np.mean(gen_slice, axis=(0, 1))

        new_avr = (im_avg + gen_avg) / 2
        gen_slice[:, :] = new_avr

        curr_err = mean_squared_error(im_slice, gen_slice)
        if curr_err < min_err:
            res = x_0, y_0, width, height
    return res


def main_opencv(args=None):
    if args is None:
        return
    print(args.image)
    fig = plt.figure(figsize=(20, 20))
    # Display input image
    ax1 = fig.add_subplot(2, 1, 1)
    im = cv.imread(args.image)  # default image channel order:
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    ax1.imshow(im)
    # plt.show(block=False)

    # Create empty image
    ax2 = fig.add_subplot(2, 1, 2)
    new_im = np.zeros(im.shape, dtype=np.int)
    new_avg = np.mean(im, axis=(0, 1))
    new_im[:, :, 0] = new_avg[0]
    new_im[:, :, 1] = new_avg[1]
    new_im[:, :, 2] = new_avg[2]
    ax2.imshow(new_im)

    for i in range(args.n_iter):
        print("iteration", i, "out of", args.n_iter)
        decay = args.rect_fraction_decay ** i
        max_w = max(im.shape[0] * args.rect_fraction * decay, 10)
        max_h = max(im.shape[1] * args.rect_fraction * decay, 10)
        print("iteration", i, "out of", args.n_iter, ", decay", decay)
        if args.population_size == 1:
            x_r, y_r, width, height = random_rect(max_x=im.shape[0], max_y=im.shape[1],
                                                  max_w=max_w, max_h=max_h)
        else:
            x_r, y_r, width, height = suggest_next(image=im, gen_image=new_im, max_x=im.shape[0],
                                                   max_y=im.shape[1], max_w=max_w, max_h=max_h,
                                                   population_size=args.population_size)
        prev_avg = np.mean(new_im[x_r:x_r + width, y_r:y_r + height, :], axis=(0, 1))
        new_avg = np.mean(im[x_r:x_r + width, y_r:y_r + height, :], axis=(0, 1))
        set_avr = (prev_avg + new_avg) / 2
        new_im[x_r:x_r + width, y_r:y_r + height, 0] = set_avr[0]
        new_im[x_r:x_r + width, y_r:y_r + height, 1] = set_avr[1]
        new_im[x_r:x_r + width, y_r:y_r + height, 2] = set_avr[2]

        if i % 100 == 0:
            ax2.imshow(new_im)
            fig.canvas.draw()
            fig.canvas.flush_events()

    plt.show()
    input("Press anything to exit")


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("-i", "--image", required=True)
    parser.add_argument("-n", "--n-iter", default=100, type=int)
    parser.add_argument("-f", "--rect-fraction", default=0.4)
    parser.add_argument("-d", "--rect-fraction-decay", default=0.999, type=float)
    parser.add_argument("-p", "--population-size", default=10, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    plt.ion()
    args = get_arguments()
    main_opencv(args=args)
