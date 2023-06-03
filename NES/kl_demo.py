import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.widgets import Slider
  
x = np.linspace(-40, 40, 100)
n_samples = 4
sample_x = np.linspace(-20, 20, n_samples)
fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)

def redraw():
    ax.cla()
    ax.figure.set_size_inches(10, 8)
    ax.set_ylim([0, 0.05])
    ax.plot(x, norm.pdf(x, m0_slider.val, s0_slider.val), 'b-')
    ax.plot(x, norm.pdf(x, m1_slider.val, s1_slider.val), 'y-')
    ax.vlines(x = sample_x, ymin = 0, ymax = 0.05, colors = 'green', linestyles='dashed')
    sample_p = norm.pdf(sample_x, m0_slider.val, s0_slider.val)
    sample_q = norm.pdf(sample_x, m1_slider.val, s1_slider.val)
    ax.plot(sample_x, sample_p, 'bo')
    ax.plot(sample_x, sample_q, 'yo')
    for i in range(n_samples):
        P_text = f'P(x)\n={(sample_p[i]):.3f}'
        ax.text(sample_x[i]-6, sample_p[i], P_text)
    for i in range(n_samples):
        Q_text = f'Q(x)\n={(sample_q[i]):.3f}'
        ax.text(sample_x[i]-6, sample_q[i], Q_text)
    for i in range(n_samples):
        ratio_text = f'log(P(x)/Q(x))\n={np.log2((sample_p[i]/sample_q[i])):.2f}'
        ax.text(sample_x[i]-5, 0.051, ratio_text)
    kl_text = 'KL(P,Q) ~='
    kl_value = 0
    for i in range(n_samples-1):
        kl_text += f'{sample_p[i]:.3f}*{np.log2((sample_p[i]/sample_q[i])):.2f} + '
        kl_value += sample_p[i] * np.log2((sample_p[i]/sample_q[i]))
    kl_text += f'{sample_p[n_samples-1]:.3f}*{np.log2((sample_p[n_samples-1]/sample_q[n_samples-1])):.2f} = {kl_value:.2f}'
    ax.text(-30, 0.056, kl_text)
    ax.set_xlabel('x')
    ax.set_ylabel('P(x) and Q(x)')
    plt.draw()

m0_ax = fig.add_axes([0.2, 0.1, 0.65, 0.03])
m0_slider = Slider(m0_ax, 'mean0', -10, 10, valinit=-10)
m0_slider.on_changed(lambda _: redraw())
s0_ax = fig.add_axes([0.2, 0.075, 0.65, 0.03])
s0_slider = Slider(s0_ax, 'std0', 8, 12, valinit=9)
s0_slider.on_changed(lambda _: redraw())
m1_ax = fig.add_axes([0.2, 0.05, 0.65, 0.03])
m1_slider = Slider(m1_ax, 'mean1', -10, 10, valinit=5)
m1_slider.on_changed(lambda _: redraw())
s1_ax = fig.add_axes([0.2, 0.025, 0.65, 0.03])
s1_slider = Slider(s1_ax, 'std1', 8, 12, valinit=12)
s1_slider.on_changed(lambda _: redraw())

redraw()
plt.show()