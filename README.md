# An Offline handwriting recognition project
This project is intended to be an offline handwriting recognition system, developed for the purpose of learning and exploration. The project is in progress, however, the current version runs well enough for simple handwritten words prediction, and retraining on the IAM dataset.

## how to use the project

### requirements 
to install the requirements pip can be used, as per: 
```
$ pip install -r requirements.txt

```
### prediction
Prediction with this project is quite straightforward, just use the help argument in the CLI for guidance by using: 
```
$ python src/predict.py -h

```
e.g:  <img align = "left" src="https://user-images.githubusercontent.com/73838152/126669479-584ba4a1-baa0-424d-8918-e78880a21d0c.jpeg" width="175" />

```
$ python src/predict.py 'img.png' -v
The detected word is: from
The time it took to predict: 2.87
```
### training
If you wish to retrain the models. ensure that the IAM words dataset is downloaded as is and words.txt is in the same directory as the /words directory, then input the path to the directory as per the guidance in:
```
$ python src/train.py -h
```

## The models used
This project was heavily inspired by the following article [Build a Handwritten Text Recognition System using TensorFlow](https://towardsdatascience.com/2326a3487cd5) , and the conventional CNN that was added as an option is a version of the CNN used in the article. However, the main model used in this program was a model inspired by the inception structure, which seemed to improve the loss immensely:


<p align="center">
<img src="https://user-images.githubusercontent.com/73838152/126667314-17f75bfe-163a-417f-b48d-571bcd02ee6d.png" width="600" />
</p>

## To Do
- [ ] Exception handling and testing. 
- [ ] More logging.
- [ ] Introduce more datasets, and generic importation methods.
- [ ] A training model for detecting words on a page.
- [ ] A GUI for ease of use.
 
