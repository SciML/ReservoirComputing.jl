# Generative vs Predictive

The library provides two different methods for prediction, denoted as `Predictive()` and `Generative()`. These methods correspond to the two major applications of Reservoir Computing models found in the literature. This section aims to clarify the differences between these two methods before providing further details on their usage in the library.

## Predictive

In the first method, users can utilize Reservoir Computing models in a manner similar to standard Machine Learning models. This involves using a set of features as input and a set of labels as outputs. In this case, both the feature and label sets can consist of vectors of different dimensions. Specifically, let's denote the feature set as ``X=\{x_1,...,x_n\}`` where ``x_i \in \mathbb{R}^{N}``, and the label set as ``Y=\{y_1,...,y_n\}`` where ``y_i \in \mathbb{R}^{M}``.

To make predictions using this method, you need to provide the feature set that you want to predict the labels for. For example, you can call `Predictive(X)` using the feature set ``X`` as input. This method allows for both one-step-ahead and multi-step-ahead predictions.

## Generative

The generative method provides a different approach to forecasting with Reservoir Computing models. It enables you to extend the forecasting capabilities of the model by allowing predicted results to be fed back into the model to generate the next prediction. This autonomy allows the model to make predictions without the need for a feature dataset as input.

To use the generative method, you only need to specify the number of time steps that you intend to forecast. For instance, you can call `Generative(100)` to generate predictions for the next one hundred time steps.

The key distinction between these methods lies in how predictions are made. The predictive method relies on input feature sets to make predictions, while the generative method allows for autonomous forecasting by feeding predicted results back into the model.
