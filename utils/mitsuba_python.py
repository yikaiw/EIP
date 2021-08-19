import os, sys
sys.path.append('/Applications/Mitsuba.app/python/2.7')
os.environ['PATH'] = 'path-to-mitsuba-directory' + os.pathsep + os.environ['PATH']
import mitsuba
from mitsuba.core import *
from mitsuba.render import SceneHandler
from mitsuba.render import RenderQueue, RenderJob
import multiprocessing

data_dir = '/Users/home/Documents/works/deformable-objects/outputs/can_tactile/'
target_dir = '/Users/home/Documents/works/deformable-objects/outputs/can_tactile/render0'
# os.makedirs(target_dir, exist_ok=True)

fileResolver = Thread.getThread().getFileResolver()
fileResolver.appendPath(data_dir)
paramMap = StringMap()
paramMap['myParameter'] = 'value'

scheduler = Scheduler.getInstance()
for i in range(0, multiprocessing.cpu_count()):
    scheduler.registerWorker(LocalWorker(i, 'wrk%i' % i))
scheduler.start()


for idx in range(7, 107):
    print('idx ' + str(idx))
    f = open(os.path.join(data_dir, 'can_origin.xml'), 'r')
    content = f.read()
    content = content.replace('elastic_voxel', '%d_elastic_voxel' % idx)
    w = open(os.path.join(data_dir, 'can.xml'), 'w')
    w.write(content)
    f.close()
    w.close()
    scene = SceneHandler.loadScene(fileResolver.resolve('can.xml'), paramMap)


    queue = RenderQueue()
    scene.setDestinationFile(os.path.join(target_dir, 'renderedResult' + str(idx)))
    job = RenderJob('myRenderJob' + str(idx), scene, queue)
    job.start()
    queue.waitLeft(0)
    queue.join()
    # print(Statistics.getInstance().getStats())
