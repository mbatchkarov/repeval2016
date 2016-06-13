import numpy as np
import pandas as pd
from matplotlib import pylab as plt
import seaborn as sns

sns.set_style("white")
sns.set_palette('cubehelix', 5)
rc = {'xtick.labelsize': 16,
      'ytick.labelsize': 16,
      'axes.labelsize': 18,
      'axes.labelweight': '900',
      'legend.fontsize': 10,
      'font.family': 'cursive',
      'font.monospace': 'Nimbus Mono L',
      'lines.linewidth': 2,
      'lines.markersize': 9,
      'xtick.major.pad': 20}
sns.set_context(rc=rc)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 20

from IPython import get_ipython

try:
    get_ipython().magic('matplotlib inline')
    plt.rcParams['figure.figsize'] = 12, 9  # that's default image size for this
    plt.rcParams['savefig.dpi'] = 2 * plt.rcParams['savefig.dpi']
except AttributeError:
    # when not running in IPython
    pass


def sparsify_axis_labels_old(ax, n=2):
    """
    Sparsify tick labels on the given matplotlib axis, keeping only those whose index is divisible by n. Works
    with factor plots
    """
    for idx, label in enumerate(ax.xaxis.get_ticklabels()):
        if idx % n != 0:
            label.set_visible(False)


def sparsify_axis_labels(ax, x=0, y=0):
    """
    Sparsify tick labels on the given matplotlib axis
    works with facet grids
    """
    from matplotlib import ticker
    if y > 0:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y))

    if x > 0:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x))


def my_bootstrap(*args, **kwargs):
    return np.vstack(args)


def tsplot_for_facetgrid(*args, **kwargs):
    """
    sns.tsplot does not work with sns.FacetGrid.map (all condition in a subplot are drawn in the same color).
    This is because either tsplot misinterprets the color parameter, or because FacetGrid incorrectly
    decides to pass in a color parameter to tsplot. Not sure which one it is, but removing that parameter
    fixes the problem
    """
    if 'color' in kwargs:
        kwargs.pop('color')
    sns.tsplot(*args, **kwargs)
