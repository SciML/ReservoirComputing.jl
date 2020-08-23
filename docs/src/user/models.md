# Models

## Echo State Network
* Constructor
```@docs
    ESN
```
* Train
```@docs
    ESNtrain
```
General function for training, used for all the ```AbstractReservoirComputers``` types with the exception of ```RMM``` which has the training method inside the constructor.

* Predict
```@docs
    ESNpredict
    ESNpredict_h_steps
```

For a full list of training and prediction methods for ESNs please refer to the User Guide ESN mods.
## Double Activation Function Echo State Network
* Constructor
```@docs
    dafESN
```
* Predict
```@docs
    dafESNpredict
    dafESNpredict_h_steps
```
## Reservoir Computing with Cellular Automata
* Constructors
```@docs
    RECA_discrete
    RECA_TwoDim
```
* Predict
```@docs
    RECAdirect_predict_discrete
    RECATDdirect_predict_discrete
    RECATD_predict_discrete
```
## Reservoir Memory Machine
* Constructor
```@docs
    RMM
```

* Predict
```@docs
    RMMdirect_predict
```
## Gated Recurrent Unit ESN
* Constructor
```@docs
    GRUESN
```
* Predict
```@docs
    GRUESNpredict
```

