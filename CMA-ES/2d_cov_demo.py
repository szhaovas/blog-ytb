import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.widgets import Slider

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
fig.subplots_adjust(left=0.25, bottom=0.3)

x = np.linspace(-2, 2, 1000)
y = np.linspace(-2, 2, 1000)
xx, yy = np.meshgrid(x, y)

def redraw_contour():
    ax.cla()
    C = np.array([[sigma_00_slider.val, sigma_01_slider.val], [sigma_01_slider.val, sigma_11_slider.val]])
    C *= scale_slider.val
    var = multivariate_normal(mean=[0,0], cov=C)
    ax.contour(
        xx, yy, var.pdf(np.dstack((xx,yy))), [0.2])
    matrix_print = (f'{C[0,0]:3.2f} {C[0,1]:3.2f}\n'
                    f'{C[1,0]:3.2f} {C[1,1]:3.2f}')
    ax.text(0.7, 1.4, matrix_print)
    D, B = np.linalg.eigh(C)
    ax.quiver([0,0], [0,0], [D[0]*B[0,0], D[1]*B[0,1]], [D[0]*B[1,0], D[1]*B[1,1]], color=['r', 'g'], scale=2)
    plt.draw()

sigma_00_ax = fig.add_axes([0.25, 0.2, 0.65, 0.03])
sigma_00_slider = Slider(sigma_00_ax, 'Var(xx)', 0, 1, valinit=0.2)
sigma_00_slider.on_changed(lambda _: redraw_contour())
sigma_01_ax = fig.add_axes([0.25, 0.15, 0.65, 0.03])
sigma_01_slider = Slider(sigma_01_ax, 'Cov(xy)', -1, 1, valinit=0)
sigma_01_slider.on_changed(lambda _: redraw_contour())
sigma_11_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
sigma_11_slider = Slider(sigma_11_ax, 'Var(yy)', 0, 1, valinit=0.2)
sigma_11_slider.on_changed(lambda _: redraw_contour())
scale_ax = fig.add_axes([0.25, 0.05, 0.65, 0.03])
scale_slider = Slider(scale_ax, 'Scale', 0, 2, valinit=1)
scale_slider.on_changed(lambda _: redraw_contour())

redraw_contour()
plt.show()
