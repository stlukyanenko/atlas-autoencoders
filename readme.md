# ATLAS autoencoders

This is a test assignment for the  HSF Atlas project at GSoC 2020. Our task was to encode the four-dimentional data from the Atlas experiment into 3-dimentional embeddings. 

## Dependencies

* [Pytorch](https://pytorch.org/get-started/locally/)
* [Dataset](link-was-temporaly-removed) - pkl fles should be unpacked directly into the repository directory.
* [Jupyter Notebook](https://jupyter.org/install.html)


## Structure
For this task an autoencoder was implemented. You can find the model of the autoencoder in the ```model.py``` file. The main training procedure is located in the ```train.py``` and all of the experimentation with comments and results can be found in the ```autoencoder_experiments.py```