import numpy as np
from itertools import product
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import matplotlib.transforms as transforms

x_min = -5
x_max = 5
fig_res = 50


def objective_function(x1, x2):
    """
    Implements Rastrigin function
    """
    # This value controls how "rugged" the objective landscape is
    # Make this value larger to make the optimization problem more difficult
    A = 1
    shift = np.array([4, -4])

    displacement = np.stack((x1, x2), axis=1) - shift
    sum_terms = np.square(displacement) - A * np.cos(2 * np.pi * displacement)

    return -(2 * A + np.sum(sum_terms, axis=1))


def confidence_ellipse(
    scaled_mean, cov, ax, n_std=3.0, facecolor="none", **kwargs
):
    """
    Adapted from https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(scaled_mean[0], scaled_mean[1])
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


class EG_Gaussian_Autodiff:
    """
    Follows a simple ask-tell interface. Here's the general workflow:
    1. Call ask() to get batch_size samples from the Gaussian distribution maintained by EG
    2. Get the objective values `objs` of the samples
    3. Call tell(objs) to make EG update the Gaussian distribution to "favor" high objs in the future
        - tell(objs) must be called after ask() so that EG knows the samples associated with `objs`
    """

    def __init__(self, init_mean, init_cov, eta, batch_size=100, seed=42):
        # Sets random seed to ensure reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        assert init_mean.ndim == 1
        self.sol_dim = init_mean.size
        self.mean = torch.tensor(init_mean, dtype=float, requires_grad=True)

        # Maintains and updates a lower-triangular matrix and computes 'cov = tril @ tril.T' to make sure it is PSD
        self.tril = torch.tensor(
            np.linalg.cholesky(init_cov).astype(float), requires_grad=True
        )
        assert self.tril.shape == (self.sol_dim, self.sol_dim)

        self.eta = eta

        # Creates a MultivariateNormal with the current mean and tril
        self.dist = torch.distributions.MultivariateNormal(
            loc=self.mean, covariance_matrix=self.tril @ self.tril.T
        )

        self.batch_size = batch_size
        self.prev_samples = None

        self.optimizer = torch.optim.Adam([self.mean, self.tril], lr=eta)

    def ask(self):
        # Remembers which solutions are sampled for later tell() update
        self.prev_samples = (
            self.dist.sample((self.batch_size,)).detach().numpy()
        )
        return self.prev_samples

    def tell(self, objs):
        # Checks ask() has been called
        assert not self.prev_samples is None

        # PyTorch stuff for computing mean and cov gradients automatically
        self.optimizer.zero_grad()
        log_probs = self.dist.log_prob(torch.Tensor(self.prev_samples))
        loss = -torch.mean(log_probs * torch.Tensor(objs))
        loss.backward()
        self.optimizer.step()

        # Recreates MultivariateNormal with updated mean and tril
        self.dist = torch.distributions.MultivariateNormal(
            loc=self.mean, covariance_matrix=self.tril @ self.tril.T
        )
        # Resets solution memory
        self.prev_samples = None


if __name__ == "__main__":
    # canvas
    x1 = np.linspace(x_min, x_max, fig_res)
    x2 = np.linspace(x_min, x_max, fig_res)
    comb = np.array(list(product(x1, x2)))
    img = (
        objective_function(comb[:, 0], comb[:, 1]).reshape((fig_res, fig_res)).T
    )

    # plot setup
    fig, ax = plt.subplots(1)
    ax.set_aspect("equal")
    plt.xlabel(rf"$x_1$")
    plt.xticks([0, fig_res - 1], [x_min, x_max])
    plt.ylabel(rf"$x_2$")
    plt.yticks([0, fig_res - 1], [x_min, x_max])

    # main loop
    eg = EG_Gaussian_Autodiff(
        init_mean=np.array([-4, 0]),
        init_cov=np.array([[2, 0], [0, 2]]),
        eta=0.1,
        batch_size=50,
    )
    try:
        itr = 0
        while True:
            samples = eg.ask()
            objs = objective_function(x1=samples[:, 0], x2=samples[:, 1])

            # imshow uses coordinates ranging [0, fig_res]
            scaled_mean = (
                (eg.mean.detach().numpy() - x_min) / (x_max - x_min) * fig_res
            )
            scaled_samples = (samples - x_min) / (x_max - x_min) * fig_res

            # Get cov = tril @ tril.T
            tril = eg.tril.detach().numpy()
            cov = tril @ tril.T

            ax.imshow(img)
            ax.set_aspect("equal")
            plt.xlabel(rf"$x_1$")
            plt.xticks([0, fig_res - 1], [x_min, x_max])
            plt.ylabel(rf"$x_2$")
            plt.yticks([0, fig_res - 1], [x_min, x_max])

            for pair in scaled_samples:
                ax.add_patch(Circle((pair[0], pair[1]), 0.5, color="yellow"))

            confidence_ellipse(
                scaled_mean=scaled_mean,
                cov=cov,
                ax=ax,
                n_std=4.0,
                edgecolor="red",
            )
            plt.draw()
            plt.pause(0.1)
            plt.cla()
            input(f"\nItr {itr}; press Enter to step\n")
            print(np.mean(objs))

            eg.tell(objs=objs)
            itr += 1

    except KeyboardInterrupt:
        quit()
