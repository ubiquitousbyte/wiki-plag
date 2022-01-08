import math
from typing import Tuple

import torch
import torch.nn as nn

import numpy as np


class Decay(nn.Module):
    """
    A time varying exponential function that artificially decreases an
    initial sigma value as time increases. 

    This is known as an exponential decay function. 

    This decay function is necessary for stochastic approximation, i.e 
    for artificially decreasing the learning rate of a model. 

    Methods
    -------
    forward(time_step: int) -> float
        Exponentially decreases the initial sigma value as a function 
        of the current time step. The decayed value is returned to the caller
    """

    def __init__(self, sigma: float, time_constant: float):
        """
        Parameters
        ----------
        sigma : float 
            The initial value of the parameter

        time_constant: float 
            Heuristic value that determines the threshold of the decay.

        Example
        -------
        With sigma = 0.1 and time_constant = 1000, the returned decay will 
        remain in the range [0.1, 0.01].  
        """

        super(Decay, self).__init__()
        self._sigma = sigma
        self._time_const = time_constant

    def forward(self, time_step: int) -> float:
        """
        Parameters
        ----------
        time_step : int 
            The current time value
        """

        return self._sigma*math.exp(-time_step/self._time_const)


class Neighbourhood(nn.Module):
    """
    The neighbourhood function of the SOM. 
    """

    def __init__(self, sigma: float, time_constant: float):
        super(Neighbourhood, self).__init__()
        self._decay = Decay(sigma, time_constant)

    def forward(self, x, y, winner, t):
        """
        Parameters
        ----------
        x : torch.FloatTensor
            The flattened grid across the x-axis
        y : torch.FloatTensor
            The flattened grid across the y-axis
        winner : Tuple[int, int]
            The coordinates of the winner neuron in the output space
        t : int
            The current time step
        """

        # Compute the lateral distances of all neurons
        # relative to the winner's coordinates
        dx = torch.pow(x - x.T[winner], 2)
        dy = torch.pow(y - y.T[winner], 2)

        # Compute the neighbourhood's normalizing constant.
        # This represents the spread of the gaussian density a.k.a
        # the size of the neighbourhood.
        # At each time step, the spread decreases.
        decay = self._decay.forward(t)

        # Map the distances onto the gaussian density.
        # Neurons closest to the winner are mapped around the peak
        # of the gaussian bell curve and thus have high probabilities.
        hx = torch.exp(torch.neg(torch.div(dx, 2*decay*decay)))
        hy = torch.exp(torch.neg(torch.div(dy, 2*decay*decay)))

        # Unflatten the grid to reform the 2-D output space
        return (hx*hy).T


class SOM(nn.Module):

    def __init__(self, grid: Tuple[int, int, int],
                 sigma: float = 1.0, learning_rate: float = 0.1):
        """
        Parameters
        ----------
        grid : Tuple[int, int, int]
            The x, y and z sizes of the grid.
            x represents the height of the lattice.
            y represents the width of the lattice 
            z represents the size of the input features 
        sigma : float 
            The initial width of the neighbourhood for a winning neuron 
        learning_rate : float
            The initial learning rate of the map 
        """

        super(SOM, self).__init__()

        x, y, z = grid
        gx, gy = torch.meshgrid(torch.arange(x), torch.arange(y))

        self._gx = nn.Parameter(gx, requires_grad=False)
        self._gy = nn.Parameter(gy, requires_grad=False)
        self._W = nn.Parameter(torch.randn(x, y, z), requires_grad=False)

        # Assume an initial learning rate of 0.1
        # If we use an initial time constant of 1000,
        # then the learning rate will begin at 0.1 and decrease
        # gradually with each time step
        # A time constant equal to 1000 ensres that the learning rate
        # will always remain above 0.01.
        self._lr_decay = Decay(learning_rate, 1000)
        self._neigh = Neighbourhood(sigma, 1000/torch.log(sigma))

    def winner(self, x):
        """
        Given an input sample x, this function returns the coordinates 
        of the neuron that is closest to the sample 

        Parameters
        ----------
        x : torch.FloatTensor
            The sample for which the closest neuron is computed
        """

        dx = torch.linalg.norm(x - self._W, ord=2, dim=-1)
        winner = np.unravel_index(torch.argmin(dx), dx.size())
        return winner

    def forward(self, x: torch.FloatTensor, t: int):
        """
        Forward pass of the SOM. 

        Step 1 (Competitive process):
        Computes the coordinates of the neuron that is closest to x.

        Step 2 (Cooperative process):
        The winner neuron and its neighouring neurons are assigned 
        high probabilities over the discrete output space. 

        Parameters
        ----------
        x : torch.FloatTensor
            The input sample to apply the forward pass on
        t : float 
            The current time step 
        """

        return self._neigh.forward(self._gx, self._gy, self.winner(x), t)

    def backward(self, x, h, t):
        """
        Backward pass of the SOM. 

        Step 1 (Adaptive process):
        The winner neuron and its neighours are pushed closer to the input.
        The probability distribution computed in the forward pass 
        is the factor which regulates the updates.

        Neurons that are not in the neighbourhood have a probability of 0,
        and therefore do not get updated.

        Conversely, neurons in the neighbourhood have probabilities > 0.
        Neurons closer to the winner get higher probabilities.
        In other words, neurons closest to the winner get a stronger push 
        towards the input. 

        Parameters
        ----------
        x : torch.FloatTensor
            The current input vector
        h : torch.FloatTensor
            The probability distribution of all neighbour neurons 
        t : int
            The current time step
        """

        alpha = self._lr_decay.forward(t)
        f = alpha*h
        self._W += torch.einsum('ij,ijk->ijk', f, x - self._W)


if __name__ == '__main__':
    som = SOM((5, 5, 10))

    x = torch.randn(1, 10)
    h = som.forward(x, 1)
    som.backward(x, h, 1)
