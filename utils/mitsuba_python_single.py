import os, sys
sys.path.append('/Applications/Mitsuba.app/python/2.7')
os.environ['PATH'] = 'path-to-mitsuba-directory' + os.pathsep + os.environ['PATH']
import mitsuba
from mitsuba.core import *
from mitsuba.render import SceneHandler
from mitsuba.render import RenderQueue, RenderJob
import multiprocessing

fileResolver = Thread.getThread().getFileResolver()
fileResolver.appendPath('/Users/home/Documents/works/deformable-objects/outputs/can_tactile/')
paramMap = StringMap()
paramMap['myParameter'] = 'value'
# Add the scene directory to the FileResolver's search path
scene = SceneHandler.loadScene(fileResolver.resolve('can.xml'), paramMap)

# print(scene)

scheduler = Scheduler.getInstance()
# Start up the scheduling system with one worker per local core
for i in range(0, multiprocessing.cpu_count()):
    scheduler.registerWorker(LocalWorker(i, 'wrk%i' % i))
scheduler.start()

queue = RenderQueue()
scene.setDestinationFile('renderedResult')
job = RenderJob('myRenderJob', scene, queue)
job.start()
queue.waitLeft(0)
queue.join()
print(Statistics.getInstance().getStats())
