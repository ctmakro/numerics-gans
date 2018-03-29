import numpy as np

# sample from mixture of gaussian
def sample(num_points):
    centers = 8 # density around 8 points

    # spread out around unit circle
    angles = np.arange(centers)/centers*2*np.pi
    xcoords = np.cos(angles)
    ycoords = np.sin(angles)

    xycoords = np.stack([xcoords, ycoords],axis=1) # shape(centers,2)
    xycoords.shape = (1,centers,2)
    gaussian = np.random.normal(size=(num_points, centers, 2)) * 0.01

    points = xycoords + gaussian
    points.shape = (centers*num_points, 2)
    return points

if __name__ == '__main__':
    from plotter import interprocess_scatter_plotter as plotter

    plotter = plotter()
    s = sample(500)
    plotter.scatter(s)
    import time
    time.sleep(5)
