'''Implement different optimisers and analyse their performance'''
import torch
import torch.optim as optim
import torch.nn as nn
from torch import Tensor
from matplotlib import pyplot as plt
from celluloid import Camera
from numpy import e, pi, sqrt
from functools import partial


class Optimiser:
    def __init__(
            self,
            model,
            inputs,
            outputs,
            params=None,
            optimiser=None,
            loss_fn=nn.MSELoss()):

        self.params = params
        self.inputs = inputs
        self.outputs = outputs
        self.model = model
        self.optimiser = optimiser
        self.loss_fn = loss_fn
        self.losses = []
        self.parameters = []
    def training_loop(self, n_epochs, learning_rate):
        for epoch in range(1, n_epochs + 1):

            if self.optimiser is None:
                if self.params.grad is not None:
                    # Zero the gradient explicitly as it accumulates
                    self.params.grad.zero_()
                # Forward pass
                predictions = self.model(self.inputs, *self.params)
            else:
                # Using built-in optimisers
                self.optimiser.zero_grad()
                predictions = self.model(self.inputs)
            # Estimation of errors
            loss = self.loss_fn(predictions, self.outputs)
            # Backward pass
            loss.backward()
            if self.optimiser is None:
                ##################
                with torch.no_grad():
                    # Gradient descent implemented from scratch
                    self.params -= learning_rate * self.params.grad
                ##################
                parameters = self.params.detach().numpy()
            else:
                ##################
                # Calculate the derivates
                self.optimiser.step()
                parameters = [float(self.model.weight.detach()), float(self.model.bias.detach())]
                ##################
            self.losses.append(float(loss))
            self.parameters.append([*parameters])

            if epoch % 1000 == 0:
                print('Epoch %d, Loss %f' % (epoch, float(loss)))
                # print('Gradient: {0}'.format(self.params.grad.detach()))
                print('Parameters: ', self.parameters[-1])

    def loss_landscape(self, number_of_points=100, bounds=[(0, 10)]):
        parameter_array = []
        dimension = self.params.size()[0]
        for index in range(dimension):
            parameter_array.append(
                    torch.linspace(*bounds[index], steps=number_of_points))

        meshes = torch.meshgrid(*parameter_array)
        loss = []
        if self.optimiser is None:
            for parameters in torch.reshape(
                    (torch.stack(meshes).T),
                    [number_of_points**dimension, dimension]):
                predictions = self.model(self.inputs, *parameters)
                loss.append(self.loss_fn(predictions, self.outputs))
        return parameter_array, loss

# User defined functions


class AckleyModel:
    '''Implementation of a non-convex function Auckley function

    d-dimensional function (usually evaluated on `[-32.768, 32.768]^d`):

        f(x) = -A exp(-B sqrt(1/d sum_{i=1}^d x_i^2)) -
            exp(1/d sum_{i=1}^d cos(c x_i)) + A + exp(1)

    f has one minimizer for its global minimum at `z_1 = (0, 0, ..., 0)` with
    `f(z_1) = 0`.'''

    def __init__(
            self, dim: int = 2, X: Tensor = None
    ) -> None:
        self.dim = dim
        self._bounds = [(-32.768, 32.768) for _ in range(self.dim)]
        self._optimal_value = 0.0
        self.a = 20
        self.b = 0.2
        self.c = 2 * pi
        self.X = X

    def generate_points(self, number_of_points: int) -> None:
        self.X = torch.FloatTensor(number_of_points, self.dim).uniform_(
            *self._bounds[0])

    def _evaluate(
            self,
            points: Tensor = None,
            a: float = None,
            b: float = None,
            c: float = None) -> Tensor:
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        if c is None:
            c = self.c
        if points is not None:
            part1 = -a * torch.exp(
                    -b / sqrt(self.dim) * torch.norm(points, dim=-1))
            part2 = -(torch.exp(torch.mean(torch.cos(c * points), dim=-1)))
        return part1 + part2 + a + e

    def evaluate_true(self, points: Tensor = None) -> Tensor:
        if points is not None:
            return self._evaluate(points)
        else:
            return self._evaluate(self.X)

    def evaluate_estimate(
            self,
            points: Tensor,
            a: float = None, b: float = None, c: float = None) -> Tensor:
        return self._evaluate(points, a, b, c)


def linear_model(t_u, w, b):
    '''Model to be established; w - weight, b - bias'''
    return w * t_u + b

# Visuallisation
def loss_landscape_3d(parameter_array, loss_array, parameter_points, loss_points):
    meshes = torch.meshgrid(*parameter_array)
    output = torch.reshape(torch.Tensor(loss_array), meshes[0].size())
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d' )
    fontsize_ = 20 # set axis label fontsize
    labelsize_ = int(0.5 * fontsize_) # set tick label fontsize
    ax.view_init(elev=30, azim=-10)
    ax.set_xlabel(r'parameter $a$', fontsize=fontsize_, labelpad=10)
    ax.set_ylabel(r'parameter $b$', fontsize=fontsize_, labelpad=10)
    ax.set_zlabel('loss', fontsize=fontsize_, labelpad=10)
    ax.tick_params(axis='x', pad=5, which='major', labelsize=labelsize_)
    ax.tick_params(axis='y', pad=5, which='major', labelsize=labelsize_)
    ax.tick_params(axis='z', pad=5, which='major', labelsize=labelsize_)
    ax.set_xlim(
        min(parameter_array[0]),
        max(parameter_array[0]))
    ax.set_ylim(
        min(parameter_array[1]),
        max(parameter_array[1]))
    ax.set_zlim(
        0,
        20)

    lag = 500
    camera=Camera(fig) # create Camera object
    for index, point in enumerate(parameter_points[::lag]):
        epoch = index + index*(lag-1)
        ax.scatter(point[0], point[1], loss_points[epoch],
                marker="o", s=100,
                color="black", alpha=1.0)
        # Surface plot (= loss landscape):
        ax.plot_surface(
                meshes[0].numpy().T, meshes[1].numpy().T, output.numpy(), cmap='terrain',
                antialiased=True, cstride=1, rstride=1, alpha=0.75)
        plt.tight_layout()
        camera.snap() # take snapshot after each iteration

    video = camera.animate(
                                repeat=False,
                                repeat_delay=0)
    video.save('optimisation.gif', dpi=300)  # save animation


if __name__ == '__main__':
    # Choose the test
    case = 'ackley'

    if case == 'ackley':
        model = AckleyModel()
        model.generate_points(1000)
        inputs = model.X
        outputs = model.evaluate_true()

        params = torch.tensor([1.0, 0.0], requires_grad=True)
        learning_rate = 1e-3

        gd = Optimiser(
                model=partial(model.evaluate_estimate, c=model.c),
                params=params,
                inputs=inputs,
                outputs=outputs)

        gd.training_loop(
            n_epochs=10000,
            learning_rate=learning_rate)
        print('Parameters obtained with GD: ', gd.params.detach().numpy())

        predicted_outputs = model.evaluate_estimate(inputs, *gd.params)

    if case == 'linear':
        # Implementing the example from "Deep-Learning-with-PyTorch"
        # temperatures in Celsius
        t_c = [
                0.5,  14.0, 15.0,
                28.0, 11.0,  8.0,
                3.0, -4.0,  6.0, 13.0, 21.0]
        # temperatures in unknown units
        t_u = [
                35.7, 55.9, 58.2,
                81.9, 56.3, 48.9,
                33.9, 21.8, 48.4, 60.4, 68.4]
        # When using NN, we need to provide a matrix of size BxNin
        t_c = torch.tensor(t_c).unsqueeze(1)
        t_u = torch.tensor(t_u).unsqueeze(1)

        # Normalise: Change the output to
        # reduce the gradient diff between w and b
        t_un = 0.1 * t_u
        params = torch.tensor([1.0, 0.2], requires_grad=True)
        learning_rate = 1e-2
        # The requires_grad=True argument
        # is telling PyTorch to track the entire
        # family tree of tensors resulting from operations on params.

        gd = Optimiser(
                model=linear_model,
                params=params,
                inputs=t_un,
                outputs=t_c)

        gd.training_loop(
            n_epochs=10000,
            learning_rate=learning_rate)
        print('Parameters obtained with GD: ', gd.params.detach().numpy())

        t_p = linear_model(t_un, *gd.params)

        # Plot the results
        fig = plt.figure()
        plt.xlabel("Temperature (°Fahrenheit)")
        plt.ylabel("Temperature (°Celsius)")
        plt.plot(t_u.numpy(), t_p.detach().numpy())
        plt.plot(t_u.numpy(), t_c.numpy(), 'o')
        plt.show()

        print('There are many optim options to choose from: ', dir(optim))
        print('Let us see how the built-in GD performs')

         # Use NN for training
        linear_model_nn = nn.Linear(1, 1)

        optimiser = Optimiser(
                model=linear_model_nn,
                inputs=t_un,
                outputs=t_c,
                optimiser=optim.SGD(
                    linear_model_nn.parameters(),
                    lr=learning_rate),
                loss_fn=nn.MSELoss())

        optimiser.training_loop(
            n_epochs=10000,
            learning_rate=learning_rate)

        print(
            'Parameters obtained with GD: ',
            optimiser.model.weight, optimiser.model.bias)
