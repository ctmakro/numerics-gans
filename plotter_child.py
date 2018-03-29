
import matplotlib
import time
import threading as th
import numpy as np

from sys import platform as _platform

if _platform == "darwin":
    # MAC OS X
    matplotlib.use('qt5Agg')
    # avoid using cocoa or TK backend, since they require using a framework build.
    # conda install pyqt

import matplotlib.pyplot as plt

class plotter:
    def __init__(self,num_lines=1):
        self.lock = th.Lock()
        self.x = []
        self.y = []
        self.num_lines = num_lines
        self.ys = [[] for i in range(num_lines)]
        self.labels = [str(i) for i in range(num_lines)]

        self.colors = [self.getcolor(i) for i in range(num_lines)]

        self.time = time.time()
        self.time_between_redraw = 1.

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)

        plt.show(block=False)
        # plt.show()

        self.something_new = True

    def getcolor(self,index):
        i = index
        return [
            (i*i*7.9+i*19/2.3+17/3.1)%0.5+0.2,
            (i*i*9.1+i*23/2.9+31/3.7)%0.5+0.2,
            (i*i*11.3+i*29/3.1+37/4.1)%0.5+0.2,
        ]

    def custom_pause(self,interval):
        # per https://stackoverflow.com/a/44352761
        # and https://stackoverflow.com/a/45734500
        # June 2017 sucks!
        # calling plot() or pause() with newer matplotlib versions always focus the plot to foreground. to prevent this from happening, you have to use alternative methods to update the plot.

        # also per https://github.com/matplotlib/matplotlib/pull/9061/commits/1f083e45fa5c022112967d2bfd966f073ffb42b0
        if self.fig.canvas.figure.stale:
            self.fig.canvas.draw()
        self.fig.canvas.start_event_loop(interval)

    def redraw(self):
        self.ax.grid(color='#f0f0f0', linestyle='solid', linewidth=1)

        x = self.x

        for idx in range(len(self.ys)): # original plot
            y = self.ys[idx]
            c = self.colors[idx]
            self.ax.plot(x,y,color=tuple(c),label=self.labels[idx])
            self.ax.legend()

        for idx in range(len(self.ys)): # low pass plot
            y = self.ys[idx]
            c = self.colors[idx]
            init = 5
            if len(y)>init:
                ysmooth = [sum(y[0:init])/init]*init
                for i in range(init,len(y)): # first order
                    ysmooth.append(ysmooth[-1]*0.9+y[i]*0.1)
                for i in range(init,len(y)): # second order
                    ysmooth[i] = ysmooth[i-1]*0.9+ysmooth[i]*0.1

                self.ax.plot(x,ysmooth,lw=2,color=tuple([cp**0.3 for cp in c]),alpha=0.5)

    def show(self):
        self.lock.acquire()
        t = time.time()
        if t - self.time > self.time_between_redraw:
            # dont redreaw if too frequent
            if self.anything_new():
                # redraw!
                self.ax.clear()

                self.redraw()

                redraw_time = time.time() - t
                self.time_between_redraw = redraw_time*2
                self.time = time.time()

        self.lock.release()
        # plt.pause(0.2)
        # plt.draw()
        self.custom_pause(0.2)

    def pushy(self,y):
        self.lock.acquire()
        self.y.append(y)
        if len(self.x)>0:
            self.x.append(self.x[-1]+1)
        else:
            self.x.append(0)
        self.something_new = True
        self.lock.release()

    def pushys(self,ys):
        self.lock.acquire()
        for idx in range(self.num_lines):
            self.ys[idx].append(ys[idx])

        if len(self.x)>0:
            self.x.append(self.x[-1]+1)
        else:
            self.x.append(0)
        self.something_new = True
        self.lock.release()

    def setlabels(self,labels):
        self.labels = labels

    def anything_new(self):
        # self.lock.acquire()
        n = self.something_new
        self.something_new = False
        # self.lock.release()
        return n

class scatterplotter(plotter):
    def __init__(self,num_lines=1):
        super().__init__(1)
        self.clearscatter()

    def scatter(self,arr):
        self.lock.acquire()
        self.something_new = True
        self.scatters.append(arr)
        self.lock.release()

    def clearscatter(self):
        self.lock.acquire()
        self.scatters = []
        self.lock.release()

    def redraw(self):
        for idx,s in enumerate(self.scatters):
            self.ax.scatter(x=s[:,0],y=s[:,1], color=[self.getcolor(idx)+[0.15]])

def stuff(plotter):
    from llll import sbc
    sbc = sbc()

    p = None
    endflag = False

    # wait for init parameters
    while 1:
        msg = sbc.recv()
        if p is None:
            if msg[0] == 'init':
                p = plotter(msg[1])
                break

    def receive_loop():
        while 1:
            msg = sbc.recv()
            if msg[0] == 'pushys':
                p.pushys(msg[1])
            elif msg[0] == 'setlabels':
                p.setlabels(msg[1])
            elif msg[0] == 'scatter':
                p.scatter(msg[1])
            elif msg[0] == 'clearscatter':
                p.clearscatter()
            else:
                break

    rt = th.Thread(target = receive_loop, daemon = True)
    rt.start()

    while 1:
        if rt.is_alive():
            p.show()
        else:
            break

if __name__ == '__main__':
    stuff(plotter)
