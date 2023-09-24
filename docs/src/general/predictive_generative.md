# Generative vs Predictive
The library provides two different methods for prediction, denoted as `Predictive()` and `Generative()`, following the two major applications of Reservoir Computing models found in the literature. Both of these methods are given as arguments for the trained model. While copy-pasteable example swill be provided further on in the documentation, it is better to clarify the difference early on to focus more on the library implementation going forward.

## Predictive
In the first method, the user can use Reservoir Computing models similarly as standard Machine Learning models. This means using a set of features as input and a set of labels as outputs. In this case, the features and labels can be vectors of different dimensions, as ``X=\{x_1,...,x_n\} \ x_i \in \mathbb{R}^{N}`` and ``Y=\{y_1,...,y_n\} \ y_i \in \mathbb{R}^{M}`` where ``X`` is the feature set and ``Y`` the label set. Given the difference in dimensionality for the prediction call, it will be needed to feed to the function the feature set to be labeled, for example by calling `Predictive(X)` using the set given in this example.

!this allows for one step ahead or h steps ahead prediction.

## Generative
The generative method allows the user to extend the forecasting capabilities of the model, letting the predicted results to be fed back into the model to generate the next prediction. By doing so, the model can run autonomously, without any feature dataset as input. The call for this model needs only the number of steps that the user intends to forecast, for example calling `Generative(100)` to generate one hundred time steps.
