import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import matplotlib.transforms as transforms

pop_size = 10
elite_size = 3
var = 4

'''
Adapted from https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
'''
def confidence_ellipse(mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def plt_show():
    confidence_ellipse(
        mean=mean,
        cov=np.eye(2)*var,
        ax=ax, n_std=4.0, edgecolor='red'
    )
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
