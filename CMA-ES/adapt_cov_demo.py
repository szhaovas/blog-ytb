import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import matplotlib.transforms as transforms

λ = 10
μ = 3
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
        cov=cov,
        ax=ax, n_std=4.0, edgecolor='red'
    )
    plt.draw()
    plt.pause(0.001)
    input()
    plt.cla()

x = np.linspace(-4, 2, 51)
y = np.linspace(-4, 2, 51)
xx, yy = np.meshgrid(x, y)
img = -(np.abs(xx - 1) + np.abs(yy - 1))
mean = np.array([15,25])
cov = np.eye(2) * var
elites = []
pop = []
fig,ax = plt.subplots(1)
ax.set_aspect('equal')
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)

while True:
    # generate λ individuals and add them to current population
    new_idvs = [np.random.multivariate_normal(mean, cov) for _ in range(λ)]
    pop += new_idvs
    ax.imshow(img)
    for pair in pop:
        ax.add_patch(Circle((pair[0],pair[1]),0.5,color='yellow'))
    for pair in elites:
        ax.add_patch(Circle((pair[0],pair[1]),0.5,color='magenta'))
    for pair in new_idvs:
        ax.add_patch(Circle((pair[0],pair[1]),0.5,color='green'))

    plt_show()

    # sort the population and discard low fitness individuals
    pop.sort(key=lambda x: img[max(min(int(x[1]),50),0), max(min(int(x[0]),50),0)], reverse=True)
    ax.imshow(img)
    for pair in pop[:λ]:
        ax.add_patch(Circle((pair[0],pair[1]),0.5,color='yellow'))
    for pair in pop[λ:]:
        ax.add_patch(Circle((pair[0],pair[1]),0.5,color='red'))
    pop = pop[:λ]

    plt_show()

    ax.imshow(img)
    for pair in pop[:λ]:
        ax.add_patch(Circle((pair[0],pair[1]),0.5,color='yellow'))

    plt_show()

    # get the top-μ elites and update the mean
    elites = pop[:μ]
    dif = np.array(elites) - mean
    cov = (dif.T @ dif) / μ
    mean = np.mean(elites, axis=0)
    ax.imshow(img)
    for pair in elites:
        ax.add_patch(Circle((pair[0],pair[1]),0.5,color='magenta'))
    for pair in pop[μ:]:
        ax.add_patch(Circle((pair[0],pair[1]),0.5,color='yellow'))

    plt_show()
