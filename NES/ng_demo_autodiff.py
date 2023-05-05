import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import matplotlib.transforms as transforms
import tensorflow as tf
import tensorflow_probability as tfp

λ = 100
η = 0.8

# canvas
x = np.linspace(-4, 2, 51)
y = np.linspace(-4, 2, 51)
xx, yy = np.meshgrid(x, y)
img = -(np.abs(xx - 1) + np.abs(yy - 1))

# plot setup
fig,ax = plt.subplots(1)
ax.set_aspect('equal')
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)

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

def update_plot(mean, tril, samples):
    ax.imshow(img)
    for pair in samples:
        ax.add_patch(Circle((pair[0],pair[1]),0.5,color='yellow'))
    
    tril_np = tril.numpy()
    cov = tril_np @ tril_np.T
    confidence_ellipse(
        mean=mean,
        cov=cov,
        ax=ax, n_std=4.0, edgecolor='red'
    )
    plt.draw()
    plt.pause(0.001)
    input()
    plt.cla()

def get_fitness(samples): 
    return [img[max(min(int(s[1]),50),0), max(min(int(s[0]),50),0)] for s in samples]

mean = tf.Variable([15, 25], dtype=np.float32)
tril = tf.Variable(np.eye(2) * 2, dtype=np.float32)
mvn = tfp.distributions.MultivariateNormalTriL(loc=mean, scale_tril=tril)
optimizer = tf.optimizers.Adam(learning_rate=η)
# optimizer = tf.keras.optimizers.SGD(learning_rate=η)
while True:
    samples = mvn.sample(λ)
    # This step is needed to overcome a bug with GradientTape, see issue https://github.com/tensorflow/probability/issues/999
    samples = tf.identity(samples)
    fitnesses = get_fitness(samples)
    with tf.GradientTape(persistent=True) as tape:
        loss = -tf.reduce_mean(mvn.log_prob(samples) * fitnesses)
    gradients = tape.gradient(loss, [mean, tril])
    update_plot(mean, tril, samples)
    optimizer.apply_gradients(zip(gradients, [mean, tril]))