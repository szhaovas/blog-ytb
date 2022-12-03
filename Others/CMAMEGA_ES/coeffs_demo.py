import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.widgets import Slider

fig, ax = plt.subplots()
ax.set_aspect('equal')
fig.subplots_adjust(bottom=0.2)

x = np.linspace(-6, 4, 100)
y = np.linspace(-6, 4, 100)
xx, yy = np.meshgrid(x, y)

has_colorbar = False
def redraw():
    global has_colorbar
    ax.cla()
    img = c0_slider.val * xx**2 + c1_slider.val * yy**2
    gradient = [2*c0_slider.val*(-1), 2*c1_slider.val*(-1)]
    c = ax.pcolormesh(x, y, img, cmap='RdBu', vmin=-20, vmax=20)
    ax.quiver(-1, -1, gradient[0], gradient[1], color='r', scale=2)
    func = f'z = {c0_slider.val:.2f}x^2 + {c1_slider.val:.2f}y^2'
    ax.text(-6, 4.5, func)
    grad = f'gradient = [{gradient[0]:.2f}, {gradient[1]:.2f}]'
    ax.text(0, 4.5, grad)
    ax.axis([-6, 4, -6, 4])
    if not has_colorbar:
        fig.colorbar(c, ax=ax)
        has_colorbar = True
    plt.draw()

c0_ax = fig.add_axes([0.2, 0.1, 0.65, 0.03])
c0_slider = Slider(c0_ax, 'c0', -1, 1, valinit=0.2)
c0_slider.on_changed(lambda _: redraw())
c1_ax = fig.add_axes([0.2, 0.05, 0.65, 0.03])
c1_slider = Slider(c1_ax, 'c1', -1, 1, valinit=0.2)
c1_slider.on_changed(lambda _: redraw())

redraw()
plt.show()
