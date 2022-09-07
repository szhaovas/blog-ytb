import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import matplotlib.transforms as transforms

pop_size = 10
elite_size = 3
var = 4

def plt_show():
    plt.draw()
    plt.pause(0.001)
    input()
    plt.cla()

x = np.linspace(-2, 0, 51)
y = np.linspace(-3, 4.5, 51)
xx, yy = np.meshgrid(x, y)
img = np.power(xx,3) - 10*xx + np.power(yy,3) - 10*yy
mean = np.array([15,25])
pop = [np.random.normal(mean,var) for _ in range(pop_size)]
fig,ax = plt.subplots(1)
ax.set_aspect('equal')
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)

while True:
    ax.imshow(img)
    for pair in pop:
        ax.add_patch(Circle((pair[0],pair[1]),0.5,color='yellow'))

    plt_show()

    pop.sort(key=lambda x: img[int(x[1]), int(x[0])], reverse=True)
    elites = pop[:elite_size]
    ax.imshow(img)
    for pair in elites:
        ax.add_patch(Circle((pair[0],pair[1]),0.5,color='red'))
    for pair in pop[elite_size:]:
        ax.add_patch(Circle((pair[0],pair[1]),0.5,color='yellow'))

    plt_show()

    ax.imshow(img)
    for pair in elites:
        ax.add_patch(Circle((pair[0],pair[1]),0.5,color='red'))

    plt_show()

    mean = np.mean(elites, axis=0)
    pop[elite_size:] = [np.random.normal(mean,var) for _ in range(pop_size-elite_size)]
    ax.imshow(img)
    for pair in elites:
        ax.add_patch(Circle((pair[0],pair[1]),0.5,color='red'))
    for pair in pop[elite_size:]:
        ax.add_patch(Circle((pair[0],pair[1]),0.5,color='green'))

    plt_show()
