# PyOpenPose
Python wrapper for openpose.
Library is still being develope. Code is runnable. Feel free to use in the meanwhile. 

## Features  
* Temporal tracking (Simple)
* Numpy implementation.  
* Interactive Plotters using opencv backend
* Threading for speeding up skeleton computations



# Quick Start  
Fill ```pyopenpose/openpose_config.json``` with the paths to model and python wrapper library.
# Package distribution
OPose class in ```core.py``` is the an openpose wrapper of the official python implementation.

Once a folder of skeletons have been generated in json format, you can use ```Skeleton``` class to automatically read 
and generate a ```np.ndarray```

This library allows to dinamically plot skeletons based on an adjacency matrix. 