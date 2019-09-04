# Topic extraction using LDA based model.

This works shows the approch used to solve the topic extraction problem using an unsupervising machine learning model based on Latent Dirichlet Allocation (LDA).

## Install

### Dependencies

You will need the following dependencies:
- python 3
- pip3. To install the needed modules.
- anaconda or miniconda. (Optional for environment creation and management)

### Requirements

To install the requirements run:

pip3 install -r requirements.txt

## Descriptions

### Datasets

To train this model, three datasets were used. All three were downloaded from [Kaggle](https://kaggle.com) website.

TED talks: Transcription of some TED speechs.
see: https://www.kaggle.com/rounakbanik/ted-talks

Topics: A dataset for topics extraction.
see: https://www.kaggle.com/luisfredgs/topics-classification

### Models

#### Extended Model
The extended model was the last trained model. The params adjusted to improve accuracy.
See: [Model](./models)

#### Auxiliary Models

The models on the [Auxiliary Models Folder](./models/auxiliaryModels) were obtained from the progressives developments of the algorithms.
- LDAModelBBC: Were obtained using the BBC's articles dataset.
- LDAModelBBC_TED: Were obtained using the BBC's articles and the TED talks datasets.
- LDAModelBBC_TED_Topics: Were obtained using the BBC's articles, TED talks and the Topics datasets.

## Demo

### Test

Test inference using the trained model from [model](./models) folder.

On a python IDE or terminal run:

python run.py  --document Document.txt --model models/LDAmodelExtended.pkl --dictionary models/dictionary.dict 

### Customized demo

To run a customized demo of the algorith you will chage the inputed params:
- --document: to input a different document to infer.
- --model: to input a different trained model
- --dictionary: to input a different dictionary.

In case of a non informed param the standard params will be assumed, wich is the same as the Demo Test example:
- --document: Document.txt
- --model: models/LDAmodelExtended.pkl
- --dictionary: models/dictionary.dict
