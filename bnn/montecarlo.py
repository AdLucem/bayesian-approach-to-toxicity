from scipy.stats import norm, multivariate_normal, uniform
import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt


def scatterplot(ax, X, Y, **kwargs):
  """Draw a scatterplot on the given axes"""

  ax.scatter(X, Y, **kwargs)
  return ax

def trajectory(ax, X, Y, **kwargs):
  """Draw a line plot on the given axes"""

  ax.plot(X, Y, **kwargs)
  return ax


def sample_multivariate_normal(n, mean, cov):
  """Get n samples from a normal distribution with mean, cov"""

  return multivariate_normal.rvs(mean=mean, cov=cov, size=n)

def sample_univariate_normal(n, mean, stddev):
  """get n samples from a univariate normal distribution"""

  return norm.rvs(loc=mean, scale=stddev, size=n)

def sample_uniform():
  """Get uniform sample between [0, 1]"""

  return uniform.rvs()


def run_MH(n, init, tau):

  zn = init
  samples = [init]
  acceptances = 0

  for i in range(n):

    z_new = multivariate_normal.rvs(mean=zn, cov=tau)

    if accept(zn, z_new, tau):
      samples.append(z_new)
      zn = z_new
      acceptances += 1
    else:
      samples.append(zn)

  acceptance_rate = acceptances / len(samples)
  return samples, acceptance_rate

    
