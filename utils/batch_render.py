import os
import argparse
import imageio

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True, type=str)
args = parser.parse_args()

for file in sorted(os.listdir(os.path.join('outputs', args.name, 'particles'))):
    i = int(file.replace('.bin', ''))
    print('frame %d' % i, flush=True)
    os.system('python utils/render.py --name %s --frame %d --imshow 0' % (args.name, i))
