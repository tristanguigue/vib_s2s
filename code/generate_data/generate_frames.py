"""
Animation of Elastic collisions with Gravity

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
import ffmpy
import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation

NB_CIRCLES = 3
NB_TRIANGLES = 2
SIZE = 0.08
FPS = 10


class ParticleBox:
    """Orbits class

    init_state is an [N x 4] array, where N is the number of particles:
       [[x1, y1, vx1, vy1],
        [x2, y2, vx2, vy2],
        ...               ]

    bounds is the size of the box: [xmin, xmax, ymin, ymax]
    """
    def __init__(self,
                 init_tstate,
                 init_cstate,
                 bounds=[-2, 2, -2, 2],
                 size=0.04,
                 M=0.05,
                 G=19.8):
        self.init_tstate = np.asarray(init_tstate, dtype=float)
        self.init_cstate = np.asarray(init_cstate, dtype=float)
        self.tM = M * np.ones(self.init_tstate.shape[0])
        self.cM = M * np.ones(self.init_cstate.shape[0])
        self.size = size
        self.tstate = self.init_tstate.copy()
        self.cstate = self.init_cstate.copy()
        self.time_elapsed = 0
        self.bounds = bounds
        self.G = G
        self.triangles_time = 0
        self.circles_time = 0

    def reset(self, init_tstate, init_cstate):
        self.init_tstate = np.asarray(init_tstate, dtype=float)
        self.init_cstate = np.asarray(init_cstate, dtype=float)
        self.tstate = self.init_tstate.copy()
        self.cstate = self.init_cstate.copy()
        self.time_elapsed = 0
        self.triangles_time = 0
        self.circles_time = 0

    def step(self, dt):
        """step once by dt seconds"""
        self.time_elapsed += dt

        # update positions
        self.tstate[:, :2] += dt * self.tstate[:, 2:]
        self.cstate[:, :2] += dt * self.cstate[:, 2:]

        # # find pairs of particles undergoing a collision
        # D = squareform(pdist(self.tstate[:, :2] + self.cstate[:, :2]))
        # ind1, ind2 = np.where(D < 2 * self.size)
        # unique = (ind1 < ind2)
        # ind1 = ind1[unique]
        # ind2 = ind2[unique]

        # # update velocities of colliding pairs
        # for i1, i2 in zip(ind1, ind2):
        #     # mass
        #     m1 = self.M[i1]
        #     m2 = self.M[i2]

        #     # location vector
        #     r1 = self.state[i1, :2]
        #     r2 = self.state[i2, :2]

        #     # velocity vector
        #     v1 = self.state[i1, 2:]
        #     v2 = self.state[i2, 2:]

        #     # relative location & velocity vectors
        #     r_rel = r1 - r2
        #     v_rel = v1 - v2

        #     # momentum vector of the center of mass
        #     v_cm = (m1 * v1 + m2 * v2) / (m1 + m2)

        #     # collisions of spheres reflect v_rel over r_rel
        #     rr_rel = np.dot(r_rel, r_rel)
        #     vr_rel = np.dot(v_rel, r_rel)
        #     v_rel = 2 * r_rel * vr_rel / rr_rel - v_rel

        #     # assign new velocities
        #     self.state[i1, 2:] = v_cm + v_rel * m2 / (m1 + m2)
        #     self.state[i2, 2:] = v_cm - v_rel * m1 / (m1 + m2)

        # check for crossing boundary
        crossed_x1 = (self.tstate[:, 0] < self.bounds[0] + self.size)
        crossed_x2 = (self.tstate[:, 0] > self.bounds[1] - self.size)
        self.triangles_time += self.tstate.shape[0] - np.sum(crossed_x1 | crossed_x2)

        crossed_y1 = (self.tstate[:, 1] < self.bounds[2] + self.size)
        crossed_y2 = (self.tstate[:, 1] > self.bounds[3] - self.size)

        self.tstate[crossed_y1, 1] = self.bounds[2] + self.size
        self.tstate[crossed_y2, 1] = self.bounds[3] - self.size

        self.tstate[crossed_y1 | crossed_y2, 3] *= -1

        crossed_x1 = (self.cstate[:, 0] < self.bounds[0] + self.size)
        crossed_x2 = (self.cstate[:, 0] > self.bounds[1] - self.size)
        self.circles_time += self.cstate.shape[0] - np.sum(crossed_x1 | crossed_x2)

        crossed_y1 = (self.cstate[:, 1] < self.bounds[2] + self.size)
        crossed_y2 = (self.cstate[:, 1] > self.bounds[3] - self.size)

        self.cstate[crossed_y1, 1] = self.bounds[2] + self.size
        self.cstate[crossed_y2, 1] = self.bounds[3] - self.size

        self.cstate[crossed_y1 | crossed_y2, 3] *= -1

        # add gravity
        self.tstate[:, 3] -= self.tM * self.G * dt
        self.cstate[:, 3] -= self.cM * self.G * dt


#------------------------------------------------------------
# set up initial state
init_tstate = -0.5 + np.random.random((NB_TRIANGLES, 4))
init_tstate *= 3.9
init_cstate = -0.5 + np.random.random((NB_CIRCLES, 4))
init_cstate *= 3.9

types = ['o'] * NB_CIRCLES + ['^'] * NB_TRIANGLES

box = ParticleBox(init_tstate, init_cstate, size=SIZE)
dt = 1. / FPS # 30fps


#------------------------------------------------------------
# set up figure and animation
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-3.2, 3.2), ylim=(-2.4, 2.4))

# particles holds the locations of the particles
triangles, = ax.plot([], [], '^', ms=6)
circles, = ax.plot([], [], 'o', ms=6)

# rect is the box edge
rect = plt.Rectangle(box.bounds[::2],
                     box.bounds[1] - box.bounds[0],
                     box.bounds[3] - box.bounds[2],
                     ec='none', lw=0, fc='none')
ax.add_patch(rect)

def init():
    """initialize animation"""
    global box, rect
    init_tstate = -0.5 + np.random.random((NB_TRIANGLES, 4))
    init_tstate *= 3.9
    init_cstate = -0.5 + np.random.random((NB_CIRCLES, 4))
    init_cstate *= 3.9
    box.reset(init_tstate, init_cstate)
    triangles.set_data([], [])
    circles.set_data([], [])
    rect.set_edgecolor('none')
    return triangles, circles, rect

def animate(i):
    """perform animation step"""
    global box, rect, dt, ax, fig
    box.step(dt)

    ms = int(fig.dpi * 2 * box.size * fig.get_figwidth()
             / np.diff(ax.get_xbound())[0])
    
    # update pieces of the animation
    rect.set_edgecolor('k')
    triangles.set_data(box.tstate[:, 0], box.tstate[:, 1])
    circles.set_data(box.cstate[:, 0], box.cstate[:, 1])

    triangles.set_markersize(ms)
    circles.set_markersize(ms)

    return triangles, circles, rect

targets = []
for i in range(10):
    ani = animation.FuncAnimation(fig, animate, frames=60, repeat=False,
                                  interval=100, blit=True, init_func=init)
    ani.new_frame_seq()

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html

    # plt.show()
    name = 'particle_box_' + str(i)
    ani.save(name + '.mp4', dpi=10)
    try:
        os.mkdir(name)
    except:
        pass
    ff = ffmpy.FFmpeg(inputs={name + '.mp4': None}, outputs={name + '/%04d.png': ['-pix_fmt yuv420p']})
    print(ff.cmd)
    ff.run()

    targets.append(np.array([box.circles_time, box.triangles_time]))

np.save('targets.npy', np.array(targets))