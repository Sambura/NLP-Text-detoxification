# Text detoxification 

This repo contains code to train and use a text-to-text model for transforming toxic style text into a neutral one

## Author
Kirill Samburskiy (k.samburskiy@innopolis.university), B20-RO-01

## Structure
### Notebooks
Contains data exporation notebook and a demo notebook

### References
Contains a document describing the materials used for this work

### Reports
Contains 2 report files describing this work

### Src
Source code in python for data loading, preprocessing, model training, prediction and evaluation

## Model weights
You can get the model weights from [Releases](https://github.com/Sambura/NLP-Text-detoxification/releases). `t5_detoxifier` contains weights for detoxification model, and `t5-toxicity-regressor` - for toxicity regressor, if you need it.

## How to use this repo

### Using demo notebook on Google colab
Follow this [link](https://colab.research.google.com/github/Sambura/NLP-Text-detoxification/blob/main/notebooks/2.0-demo.ipynb), or open `notebooks/2.0-demo.ipynb` on google colab. This notebook contains code that allows to quickly get and transform the data, train the model, and make some predictions. Alternatively, you can load pretrained weights. It is also possible to run this notebook locally, check `requirements.txt` for all the required packages you will need.

### Manually
Clone git repo:
```bash
git clone https://github.com/Sambura/NLP-Text-detoxification.git
cd NLP-Text-detoxification
```
Install all the packages from `requirements.txt`, according to the python distribution you are using. For example it can be done with:
```bash
pip install -r /path/to/requirements.txt
```

Now, the easiest way to train the model is to run `src/models/train_model.py`, which will start model training with the standard parameters.

For predictions use `src/models/predict_model.py`. Running without arguments this will attempt to load the model from `models/t5_detoxifier-10/` and run prediction on the whole dataset (`reference` column). To translate a single line of text you can use command line argument like so:

```bash
python ./src/models/predict_model.py -t 'Hello there general Kenobi'
```
For more info you can use `--help` command line argument.

You can also use `src/models/evaluate_model.py` to evaluate model's predictions, and `src\models\toxicity_classifiers\t5_toxicity_regressor\train_model.py` to train the toxicity regressor. Both of these scripts have a minimal command line interface, which you can query by using `--help` argument.
