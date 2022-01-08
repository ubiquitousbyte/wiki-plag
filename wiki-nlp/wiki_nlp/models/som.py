from typing import Dict, Tuple
import math

import torch
import torch.nn as nn

import numpy as np
from numpy.random import default_rng


class Decay(nn.Module):
    """
    A time varying exponential function that artificially decreases an
    initial sigma value as time increases. 

    This is known as an exponential decay function. 

    This decay function is necessary for stochastic approximation, i.e 
    for artificially decreasing the learning rate of a model that does not 
    use gradient descent.

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


class QuantizationLoss(nn.Module):
    """
    Loss metric for a SOM. 

    This loss sums the distances between the neurons of a SOM and a set of 
    input data points. 

    A large value indicates that the mapping between the input space 
    and the output space is disproportionate 

    """

    def __init__(self):
        super(QuantizationLoss, self).__init__()

    def _weight_dist(self, x: torch.FloatTensor, w: torch.FloatTensor):
        wf = w.reshape(-1, w.size(dim=2))
        xsq = torch.sum(torch.pow(x, 2), dim=1, keepdim=True)
        wfsq = torch.sum(torch.pow(wf, 2), dim=1, keepdim=True)
        ct = torch.matmul(x, wf.T)
        return torch.sqrt(-2 * ct + xsq + wfsq.T)

    def _quantize(self, x, w):
        winners = torch.argmin(self._weight_dist(x, w), dim=1)
        return w[np.unravel_index(winners, w.size()[:2])]

    def forward(self, x, w):
        return torch.mean(torch.norm(x-self._quantize(x, w), dim=1))


class SOM(nn.Module):

    def __init__(self, grid: Tuple[int, int, int],
                 sigma: float = 0.8, learning_rate: float = 0.1):
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
        gx, gy = torch.meshgrid(torch.arange(
            x), torch.arange(y), indexing='xy')

        self._gx = nn.Parameter(gx, requires_grad=False)
        self._gy = nn.Parameter(gy, requires_grad=False)

        self._W = nn.Parameter(torch.randn(x, y, z), requires_grad=False)
        self._W /= torch.linalg.norm(self._W, ord=2, dim=-1, keepdim=True)

        # Assume an initial learning rate of 0.1
        # If we use an initial time constant of 1000,
        # then the learning rate will begin at 0.1 and decrease
        # gradually with each time step
        # A time constant equal to 1000 ensres that the learning rate
        # will always remain above 0.01.
        self._lr_decay = Decay(learning_rate, 1000)

        self._neigh = Neighbourhood(sigma, 1000/math.log(sigma))

        self._rng = default_rng()

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

        # Compute the current value of the learning rate w.r.t the current time
        alpha = self._lr_decay.forward(t)
        # Truncate the probabilities by the current learning rate.
        f = alpha*h
        # Update the neurons
        self._W += torch.einsum('ij,ijk->ijk', f, x - self._W)

    def quantization_error(self, x: torch.FloatTensor):
        loss = QuantizationLoss()
        return loss.forward(x, self._W)

    def fit(self, x: torch.FloatTensor, epochs: int = 500) -> Dict[int, Tuple[int, int]]:
        """
        Trains a SOM on the data matrix

        Parameters
        ----------
        x : torch.FloatTensor
            The data matrix to train the algorithm on
        """

        # Create index sequence
        # If the number of samples in x is 100, this yiels a
        # tensor of indices [0...99]
        indices = torch.arange(x.size(dim=0))

        # Repeat the indices by the number of epochs
        # Following the example above, if epochs is 10, this
        # yields a tensor holding 10 repetitions of [0..99]
        iterations = indices.repeat(epochs)

        # Shuffle the indices
        iterations = iterations[torch.randperm(iterations.size(dim=0))]

        som = {}

        # Initiate training procedure
        for time_step, iteration in enumerate(iterations):
            # Forward pass
            h = self.forward(x[iteration], time_step)
            # Backward pass
            self.backward(x[iteration], h, time_step)
            som[iteration.item()] = self.winner(x[iteration])

        # Print quantization error
        print("\nQuantization error: ", self.quantization_error(x))

        return som


if __name__ == '__main__':
    som = SOM((5, 5, 10))

    x = torch.randn(100, 10)
    x /= torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
    m = som.fit(x)
    print(m)
