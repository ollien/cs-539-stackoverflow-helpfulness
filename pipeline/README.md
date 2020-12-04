# Model Pipeline

This folder contains the scripts necessary to run our model.

## Data Download
 In order to run the models, you must first have the data downloaded. This can be done by [installing dolt](https://www.dolthub.com/docs/tutorials/installation/) and running the following:

```
$ dolt clone ollien/cs-539-stackoverflow-data
$ cd cs-539-stackoverflow-data
$ dolt table export AllData alldata.csv
```

You may then move `alldata.csv` to wherever is most convenient for you.

## Environment setup
Before continuing, you should also make sure you have the dependencies in our `requirements.txt` (located in the repo root) installed by doing `pip3 install -r requirements.txt`.

It is also worth noting that the neural network component of the model requires a GPU that supports CUDA. If one wishes to run the model without one (not recommended), they should remove all of the calls to `.cuda()` in `torch_embeddings.py`.

 ## Running the model

To run the model, simply do
```
$ python3 run.py --n-splits 5 --data_file path/to/alldata.csv
```
This will perform a five-fold cross validation, run all the components of the model, and then output a final accuracy value.

If you wish to re-run the model, you must delete the `processed_data` directory before continuing, or specify `--storage_dir` with another directory to use for data storage.
