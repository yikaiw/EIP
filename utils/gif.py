import os
import argparse
import imageio

parser = argparse.ArgumentParser()
parser.add_argument('--folder', default='vonmises', type=str)
args = parser.parse_args()


def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name)[300:, 50:800])
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


def main(folder):
    path = 'outputs/' + folder
    image_list = sorted(os.listdir(path))
    image_list = [os.path.join(path, filename) for filename in image_list]

    gif_name = 'outputs/%s/gif.gif' % folder
    duration = 0.1
    create_gif(image_list, gif_name, duration)


if __name__ == '__main__':
    main(args.folder)
