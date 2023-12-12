# Echo State Networks
The core component of an ESN is the `ESN` type. It represents the entire Echo State Network and includes parameters for configuring the reservoir, input scaling, and output weights. Here's the documentation for the `ESN` type:

```@docs
    ESN
```

## Variations
In addition to the standard `ESN` model, there are variations that allow for deeper customization of the underlying model. Currently, there are two available variations: `Default` and `Hybrid`. These variations provide different ways to configure the ESN. Here's the documentation for the variations:

```@docs
    Default
    Hybrid
```
The `Hybrid` variation is the most complex option and offers additional customization. Note that more variations may be added in the future to provide even greater flexibility.

## Training

To train an ESN model, you can use the `train` function. It takes the ESN model, training data, and other optional parameters as input and returns a trained model. Here's the documentation for the train function:
```@docs
    train
```
With these components and variations, you can configure and train ESN models for various time series and sequential data prediction tasks.