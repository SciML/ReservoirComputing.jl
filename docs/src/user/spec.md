# Special ESNs

These ESN are "special" in the fact that they have their own training methods. One of the future goals is to merge all the training methods into one. For all these models the constructor is the same, ```ESN```.

## Support Vector Echo State Machines
* Train 
```@docs
    SVESMtrain
```
* Predict
```@docs
    SVESM_direct_predict
    SVESMpredict
    SVESMpredict_h_steps
```

## Echo State Gaussian Processes
* Train
```@docs
    ESGPtrain
```
* Predict
```@docs
    ESGPpredict
    ESGPpredict_h_steps
```
