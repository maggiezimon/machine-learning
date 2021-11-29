## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
Working space for Python scripts implementing optimisation procedures for machine learning. Some tests are motivated by the book ["Deep-Learning-with-PyTorch"](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf).

## Technologies
Project was created with:
* Python version 3.8.8
* PyTorch version 1.9.1
* Numpy version 1.20.1
* Celluloid version 0.2.0 (animation)
* Matplotlib version 3.3.4 (visualisation)
and tested using IPython 7.22.0

## Setup
To run the project, make sure to have the libraries listed in `Technologies`. The script `compare-optimisers.py` implements the class `Optimiser` which handles the learning procedure. If the optimiser object is not provided, the class will perform optimisation using gradient descent written from scratch. To see what other options are available, run the following
```
import torch.optim as optim
print('There are many optim options to choose from: ', dir(optim))
```
The optimisation object can be initiated as
```
gd = Optimiser(
                model=linear_model,
                params=params,
                inputs=t_un,
                outputs=t_c)
```
which means that we have to provide a model function that we are trying to "learn", an initial guess for the parameters (`params`) of that model and our training inputs and outputs generated with the model for known parameters. If we want to use PyTorch optimiser, we need to pass the information about the function that we want to apply, e.g., `optim.SGD`:

```
optimiser = Optimiser(
                model=linear_model_nn,
                inputs=t_un,
                outputs=t_c,
                optimiser=optim.SGD(
                    linear_model_nn.parameters(),
                    lr=learning_rate),
                loss_fn=nn.MSELoss())
```
### Testing
At the moment, two tests are implemented in the main function: `case = 'ackley'`, and `case = 'linear'`. To study the more complex example with the visualisation, choose the `ackley` test and execute from the terminal
```
ipython -i compare_optimisers.py
parameter_array, loss = gd.loss_landscape(20, [(17, 20), (0.1, 0.6)])
loss_landscape_3d(parameter_array, loss, gd.parameters, gd.losses)
```
This will generate some visualisation:
```open optimisation.gif```
