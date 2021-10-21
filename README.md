# Kaggle Datathon: [DataHub 2.0]

## Problem Statement
Music is as much a powerful form of human expression as it is an entertainment. Over time, it has developed from the earliest calls and rhythms, into a huge variety of different genres. The high contrast between the simplicity of folk songs to the complexities of classical symphonies and the hypnotic rhythms of dance music suggests that we can effectively assign a category to each song based on various elements. For example, genres can be defined by the use of specific instruments. If the piece was being played in a certain style using orchestral instruments, then we could classify it as classical music. Similarly, if the instruments were highly distorted guitars, we would classify them as rock or heavy metal. Likewise, drum and bass use a very fast bpm and is primarily electronic.

Currently, many music aggregator applications rely on machine learning to power their recommendation engine, and curate playlists. 
> In this challenge, you are expected to develop a machine learning model with the given dataset which classifies music into genres, taking into account relevant features.

## Objective
Your goal is to predict the correct genre of each music record, given their respective features!

## Evaluation Metric
The evaluation metric for this competition is Categorization Accuracy - the percentage of predictions that are correct.

## Submission Format
> Sample_submission.csv

---
## Dataset
>Use Kaggle API command, given below, to download the dataset
```sh
>_ kaggle competitions download -c datahub-2021
```
>OR
> Use Git to clone this repository 
```sh
$ git clone https://github.com/Alpha-github/Kaggle_Competition_Datahub2021.git
```
| Files | Description |
| --------| --------|
| `train.csv` | The training set|
| `test_x.csv` | The test set |
| `Sample_submission.csv` | A sample submission file in the correct format|
| `metaData.csv` | Supplemental information about the data |

| Output File| Description|
|--------|-------|
| `submission.csv` |This files contains id of test data and its respective prediction

&nbsp;
---
## Program Description
The program involves preprocessing of data using pandas and building a predictive categorical model.

As accuracy is the key, 3 models have been built; Two models using Sci-kitLearn, GaussianNB and DecisionTreeClassifier, and the last one using Tensorflow Keras Deep Learning model. The model which gives the highest accuracy will be opted.
Replace `PATH_TO_TRAIN_CSV` and `PATH_TO_TEST_CSV` with the path of your train.csv and test.csv files.

Feel free to play with the Deep Learning Model by tweeking hyperparameters, number of layers, Optimization and Loss functions, etc.

**Important:** Beware to not overfit your model, else it won't perform well on the test dataset.
The final prediction on the Test data is stored in  `submission.csv`
**NOTE:** As training Neural Networks is hardware intensive, its better to run the model using Google Colab.

## Technology
[![Python](https://www.cupaya.com/wp-content/uploads/2017/09/python-logo.png)](https://www.python.org/)

-  An easy to pick up programming language and fun to play with.

 [![pandas](https://numfocus.org/wp-content/uploads/2016/07/pandas-logo-300.png)](https://pandas.pydata.org/) 
-  Pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language.

[![tensorflow](https://idroot.us/wp-content/uploads/2019/03/TensorFlow-logo.png)](https://www.tensorflow.org/)
-  Tensorflow Keras - The core open source library to help you develop and train ML models.

[![sklearn](https://www.analyticsvidhya.com/wp-content/uploads/2015/01/scikit-learn-logo.png)](https://scikit-learn.org/stable/)

-  Simple and efficient tools for predictive data analysis
-  Accessible to everybody, and reusable in various contexts
-  Built on NumPy, SciPy, and matplotlib

[![matplotlib](https://matplotlib.org/2.1.2/_images/sphx_glr_logos2_thumb.png)](https://matplotlib.org/)

-  Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy.

[![Numpy](https://png.pngitem.com/pimgs/s/465-4651848_numpy-python-logo-hd-png-download.png)](https://numpy.org/)

-  NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

### Support Libraries: 
 - #### Logging :
 - - Python has a built-in module logging which allows writing status messages to a file or any other output streams.

 - #### Seaborn:
 - -  Seaborn is a data visualization library built on top of matplotlib and closely integrated with pandas data structures in Python.

## Setup
##### This project was built using Windows 10 
&nbsp;
#### Install [Tensorflow] using pip
System requirements :-
> Python 3.6–3.9
> Python 3.9 support requires TensorFlow 2.5 or later.
> Python 3.8 support requires TensorFlow 2.2 or later.
#
**Important:** For more information regarding proper installation and Setting up GPU. [Click here] 
```sh
pip install tensorflow
```
#### Install [Numpy] using pip

```sh
pip install numpy
```
#### Install [Pandas] using pip

```sh
pip install pandas
```
#### Install [Numpy] using pip

```sh
pip install numpy
```
#### Install [Seaborn] using pip

```sh
pip install seaborn
```
##### This project was built on Python version 3.9
 To download python, Click on the thumbnail below to be redirected to Python Downloads page

[![Python](https://www.cupaya.com/wp-content/uploads/2017/09/python-logo.png)](https://www.python.org/downloads/)
#

## License
#
#### Public


[//]: # (Links)
[DataHub 2.0]:<https://www.kaggle.com/c/datahub-2021>
[Pandas]:<https://pandas.pydata.org/>
[Tensorflow]:<https://www.tensorflow.org/>
[Sci-kit-learn]:<https://scikit-learn.org/stable/>
[Matplotlib]:<https://matplotlib.org/>
[Numpy]:<https://numpy.org/>
[Python]:<https://www.python.org/>
[Seaborn]:<https://seaborn.pydata.org/>
[CLick here]:<https://www.tensorflow.org/install/pip>