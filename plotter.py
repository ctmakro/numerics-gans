# interactive plotter that runs in a separate process.

class interprocess_plotter:
    def __init__(self,num_lines=1):
        # super().__init__(remote_plotter_callback)
        from llll import PythonInstance
        self.pi = PythonInstance(
            'plotter_child.py',
            is_filename=True,
        )

        self.pi.send(('init',num_lines))

        # self.send(('init',num_lines))

    def pushys(self,ys):
        # self.send(('pushys', ys))
        self.pi.send(('pushys', ys))

    def setlabels(self,labels):
        self.pi.send(('setlabels',labels))

    def scatter(self, arr):
        self.pi.send(('scatter',arr))

    def clearscatter(self):
        self.pi.send(('clearscatter',))

class interprocess_scatter_plotter(interprocess_plotter):
    def __init__(self,num_lines=1):
        # super().__init__(remote_plotter_callback)
        from llll import PythonInstance
        self.pi = PythonInstance(
            'plotter_scatter_child.py',
            is_filename=True,
        )

        self.pi.send(('init',num_lines))

if __name__=='__main__':
    ip = interprocess_plotter(2)
    import math,time
    ip.setlabels(['hell','yeah'])
    for i in range(100):
        ip.pushys([math.sin(i/10), math.sin(i/10+2)])
        time.sleep(0.05)

    time.sleep(2)
    del ip # ugh..

    ip = interprocess_scatter_plotter()
    import numpy as np
    for i in range(10):
        ip.clearscatter()
        for i in range(10):
            ip.scatter(np.random.uniform(size=(100,2)))
            time.sleep(0.1)

    time.sleep(2)
    del ip
