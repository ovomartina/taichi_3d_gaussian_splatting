import numpy

import sym
from math import factorial


M = (1/6)*numpy.array([[5, 3, -3, 1],
                       [1, 3, 3, -2],
                       [0, 0, 0, 1]])
M_z = numpy.array([[2, 1, -2, 1],
                        [0, 1, 3, -2],
                        [0, 0, -1, 1]])/factorial(2)
_EPS = 1e-6


def plotCoordinateFrame(axis, T_0f, size=1, linewidth=3, name=None):
    """Plot a coordinate frame on a 3d axis. In the resulting plot,
    x = red, y = green, z = blue.

    plotCoordinateFrame(axis, T_0f, size=1, linewidth=3)

    Arguments:
    axis: an axis of type matplotlib.axes.Axes3D
    T_0f: The 4x4 transformation matrix that takes points from the frame of interest, to the plotting frame
    size: the length of each line in the coordinate frame
    linewidth: the width of each line in the coordinate frame

    Usage is a bit irritating:
    import mpl_toolkits.mplot3d.axes3d as p3
    import pylab as pl

    f1 = pl.figure(1)
    # old syntax
    # a3d = p3.Axes3D(f1)
    # new syntax
    a3d = f1.add_subplot(111, projection='3d')
    # ... Fill in T_0f, the 4x4 transformation matrix
    plotCoordinateFrame(a3d, T_0f)

    see http://matplotlib.sourceforge.net/mpl_toolkits/mplot3d/tutorial.html for more details
    """
    # \todo fix this check.
    # if type(axis) != axes.Axes3D:
    #    raise TypeError("axis argument is the wrong type. Expected matplotlib.axes.Axes3D, got %s" % (type(axis)))

    p_f = numpy.array([[0, 0, 0, 1], [size, 0, 0, 1], [
                      0, size, 0, 1], [0, 0, size, 1]]).T
    p_0 = numpy.dot(T_0f, p_f)

    X = numpy.append([p_0[:, 0].T], [p_0[:, 1].T], axis=0)
    Y = numpy.append([p_0[:, 0].T], [p_0[:, 2].T], axis=0)
    Z = numpy.append([p_0[:, 0].T], [p_0[:, 3].T], axis=0)
    try:
        axis.plot3D(X[:, 0], X[:, 1], X[:, 2], 'r-', linewidth=linewidth)
        axis.plot3D(Y[:, 0], Y[:, 1], Y[:, 2], 'g-', linewidth=linewidth)
        axis.plot3D(Z[:, 0], Z[:, 1], Z[:, 2], 'b-', linewidth=linewidth)
    except:
        axis.plot(X[:, 0], X[:, 1], 'r-', linewidth=linewidth)
        axis.plot(Y[:, 0], Y[:, 1], 'g-', linewidth=linewidth)
        axis.plot(Z[:, 0], Z[:, 1], 'b-', linewidth=linewidth)

    if name is not None:
        axis.text(X[0, 0], X[0, 1], X[0, 2], name, zdir='x')


def plot_trajectory(axis, bases, resolution=0.01, size=1, linewidth=3, name=None, color="blue", label=None, evaluate_zspline=False):  # bases: 4x3
    """Plot the 3d trajectory given the spline bases as a line

    Returns:
        _type_: _description_
    """
    time_range = numpy.arange(0, 1, resolution)

    # Define the 3d location at each step
    positions = numpy.zeros((len(time_range), 3))  # samplesx3

    
    for i, step in enumerate(time_range):
        tt = numpy.power(step, numpy.arange(0, 4))
        if evaluate_zspline:
            w = numpy.matmul(M_z, tt)
        else:
            w = numpy.matmul(M, tt)
        delta_position = bases[0, :] + w[0]*(bases[1, :] - bases[0, :]) + w[1]*(
            bases[2, :] - bases[1, :]) + w[2]*(bases[3, :]-bases[2, :])
        delta_position = numpy.expand_dims(delta_position, axis=1)
        positions[i] = delta_position.reshape((1, 3))
    axis.plot(positions[:, 0], positions[:, 1], positions[:, 2],
              color=color, linewidth=linewidth, label=label)


def plot_trajectory_2d(axis, bases, resolution=0.01, size=1, linewidth=3, name=None, color="blue", label=None,evaluate_zspline=False):
    time_range = numpy.arange(0, 1, resolution)

    positions = numpy.zeros((len(time_range), 3))  # samplesx3
    for i, step in enumerate(time_range):
        tt = numpy.power(step, numpy.arange(0, 4))
        if evaluate_zspline:
            w = numpy.matmul(M_z, tt)
        else:
            w = numpy.matmul(M, tt)
        delta_position = bases[0, :] + w[0]*(bases[1, :] - bases[0, :]) + w[1]*(
            bases[2, :] - bases[1, :]) + w[2]*(bases[3, :]-bases[2, :])
        # delta_position = bases[0] + w[0]*(sym.Pose3.from_tangent(bases[1, :]).local_coordinates(sym.Pose3.from_tangent(bases[0, :]))) + \
        #     w[1]*(sym.Pose3.from_tangent(bases[2, :]).local_coordinates(sym.Pose3.from_tangent(bases[1, :]))) + \
        #     w[2]*(sym.Pose3.from_tangent(bases[3, :]
        #                                  ).local_coordinates(sym.Pose3.from_tangent(bases[2, :])))
        delta_position = numpy.expand_dims(delta_position, axis=1)
        positions[i] = delta_position.reshape((1, 3))

    axis.plot(positions[:, 0], positions[:, 1],
              color=color, linewidth=linewidth, label=label)

    axis.grid()


def plot_trajectory_lie(axis, bases, resolution=0.01, size=1, linewidth=3, name=None, color="blue"):  # bases: 4x3
    """Plot the 3d trajectory given the spline bases as a line

    Returns:
        _type_: _description_
    """
    time_range = numpy.arange(0, 1, resolution)

    positions = numpy.zeros((len(time_range), 6))  # samplesx6

    for i, step in enumerate(time_range):
        tt = numpy.power(step, numpy.arange(0, 4))
        w = numpy.matmul(M, tt)
        # delta_position = bases[0, :] + w[0]*(bases[1, :] - bases[0, :]) + w[1]*(
        #     bases[2, :] - bases[1, :]) + w[2]*(bases[3, :]-bases[2, :])
        delta_position = bases[0] + w[0]*(sym.Pose3.from_tangent(bases[1, :]).local_coordinates(sym.Pose3.from_tangent(bases[0, :]))) + \
            w[1]*(sym.Pose3.from_tangent(bases[2, :]).local_coordinates(sym.Pose3.from_tangent(bases[1, :]))) + \
            w[2]*(sym.Pose3.from_tangent(bases[3, :]
                                         ).local_coordinates(sym.Pose3.from_tangent(bases[2, :])))

        delta_position = numpy.expand_dims(delta_position, axis=1)
        positions[i] = delta_position.reshape((1, 6))

        transform = sym.Pose3.from_tangent(
            positions[i], _EPS).to_homogenous_matrix()
        plotCoordinateFrame(axis, transform, size=size, linewidth=linewidth)
