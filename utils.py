import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from mfccUtilities import *

import torch

def heatmap(values, predicted_values):
    cm = confusion_matrix(values, predicted_values)
    fig, ax = plt.subplots(figsize=(6,6))
    ax = sns.heatmap(data=cm, fmt='.0f', xticklabels=np.unique(predicted_values).size, yticklabels=np.unique(predicted_values).size, annot = True, linewidths=0)
    ax.set_aspect("equal", "datalim")
    ax.set_ylabel("True Label", fontsize=10, weight = "bold")
    ax.set_xlabel("Predicted Label", fontsize = 10, weight = "bold")
    plt.title("La matrice de confusion")
    plt.show(fig)

def plot_audio(audio: np.ndarray, fig_title: str = 'Signal audio') -> None:
    """Visualize the audio signal.

    :param audio: the audio.
    :type audio: np.ndarray
    :param fig_title: the title of the figure, defaults to 'Signal audio'
    :type fig_title: str, optional
    """
    if torch.is_tensor(audio):
        # After data augmentation, The audio got an additional layer (dimension)
        # Get rid of it here => visualization purposes only.
        audio = np.squeeze(audio.numpy())

    plt.plot(audio)
    plt.title(fig_title)
    plt.show()


def plot_mfcc(mfcc: np.ndarray, fig_title: str = 'mfcc') -> None:
    """Visualize the mfcc matrix, or the mfcc mean vector.

    :param mfcc: the mfcc matrix, or the mfcc mean vector.
    :type mfcc: np.ndarray
    :param fig_title: the figure title, defaults to 'mfcc'
    :type fig_title: str, optional
    """
    _, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(10,5))
    ax1.plot(mfcc)
    ax2.imshow(mfcc)
    plt.title(fig_title)
    plt.show()


def plot_audio_mfcc(audio: np.ndarray, mfcc: np.ndarray) -> None:
    """Visualize the audio signal and the mfcc in the same figure.

    :param audio: audio signal.
    :type audio: np.ndarray
    :param mfcc: mfcc matrix (or vector).
    :type mfcc: np.ndarray
    """
    fig = plt.figure(figsize = (10,5))
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax1.plot(audio)
    ax1.set_title('Signal audio')

    ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    ax2.plot(mfcc)
    ax2.set_title('mfcc')

    ax3 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    ax3.imshow(mfcc)
    ax3.set_title('mfcc')
    plt.show()


def bar_plot(data_path:str) -> list:
    """Visualize the number of audio files per category.

    :param data_path: the path to the folder containing the audio files.
    :type data_path: str
    :return: the category names.
    :rtype: list
    """
    # the files within the new dataset
    fichiers = listdir(data_path)

    # the classes within the new dataset
    all_classes = []
    try:
        for fname in fichiers:
            name = fname.split('.')[0]
            label = name.split('_')[-1]
            all_classes.append(label)
    except:
        print('Exception ', sys.exc_info()[0], ' occured!')


    # plot the bars
    pd.Series(all_classes).value_counts().plot(kind = 'bar')


    # get list of unique classes
    all_classes = list(set(all_classes))
    # print('Nous avons les classes suivantes : ')
    # print(all_classes)
    return all_classes

def load_data(datapath, classes, test_percent, scale):
    """_summary_

    Args:
        datapath (_type_): _description_
        classes (_type_): _description_
        test_percent (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Get file names and sort them in order to have the same order
    # no matter what OS you are using
    files = sorted(listdir(datapath))
    files = np.array(files)

    # get the label corresponding to each file
    # and filter the files, only keep the ones with a class 
    # belonging to the list : classes
    labels = []
    files_to_keep = []
    for fname in files:
        name, extension = fname.split('.')
        name, label = name.split('_')
        if label in classes:
            files_to_keep.append(fname)
            labels.append(classes.index(label))
    labels = np.array(labels)

    sss = StratifiedShuffleSplit(n_splits=1, test_size= test_percent, random_state=0)
    for train_index, test_index in sss.split(files_to_keep, labels):
        # just to get the train_index and test_index lists
        train_index = np.array(train_index)
        test_index = np.array(test_index)

    # Formattage des labels
    LabelApp = labels[train_index]
    LabelTest = labels[test_index]

    # Calcul des MFCC sur tous les fichiers de la base d'apprentissage
    BaseApp=[]
    BaseTest=[]
    for i in range(len(train_index)):
        filepath = join(datapath, files_to_keep[train_index[i]])
        audio, sr = load_audio(filepath, scale)
        BaseApp.append((audio, sr))

    for i in range(len(test_index)):
        filepath = join(datapath, files_to_keep[test_index[i]])
        audio, sr = load_audio(filepath, scale)
        BaseTest.append((audio, sr))
    

    BaseApp=np.asarray(BaseApp)
    BaseTest=np.asarray(BaseTest)

    return BaseApp, BaseTest, LabelApp, LabelTest

def get_dataframe(datapath, keep_classes):

    # Get file names and sort them in order to have the same order
    # no matter what OS you are using
    files = sorted(listdir(datapath))

    # Initiate the data dictionnary
    df = {
        "filename" : [],
        "filepath" : [],
        "noise": [],
        "label": [],
        "labelId": []
    }

    # Determine the label, filepath and whether the file contain noise of not
    for fname in files:
        fpath = join(datapath, fname)
        noise = "bruite" in fname
        name, _ = fname.split('.')
        _, label = name.split('_')
        
        df["filename"].append(fname)
        df["filepath"].append(fpath)
        df["noise"].append(noise)
        df["label"].append(label)
        df["labelId"].append(keep_classes.index(label))

    df = pd.DataFrame(df)
    return df

def stratified_split(datapath, test_pct = 0.2, keep_classes = None):
    
    dataframe = get_dataframe(datapath, keep_classes)

    if keep_classes:
        dataframe = dataframe.loc[dataframe.label.isin(keep_classes)]
        
    sss = StratifiedShuffleSplit(n_splits=1, test_size= test_pct, random_state=0)
    for train_index, test_index in sss.split(dataframe.filename, dataframe.labelId):
        # just to get the train_index and test_index lists
        train_indices = np.array(train_index)
        test_indices = np.array(test_index)
    train_df, test_df = dataframe.iloc[train_indices], dataframe.iloc[test_indices]
    return train_df, test_df

def inference(model, val_dl, device):
  correct_prediction = 0
  total_prediction = 0
  y_pred = []
  y = []

  # Disable gradient updates
  with torch.no_grad():
    for data in val_dl:
      # Get the input features and target labels, and put them on the GPU
      inputs, labels = data[0].to(device), data[1].to(device)
      y.extend(labels)

      # Normalize the inputs
      inputs_m, inputs_s = inputs.mean(), inputs.std()
      inputs = (inputs - inputs_m) / inputs_s

      # Get predictions
      outputs = model(inputs)

      # Get the predicted class with the highest score
      _, prediction = torch.max(outputs,1)
      y_pred.extend(prediction)

      # Count of predictions that matched the target label
      correct_prediction += (prediction == labels).sum().item()
      total_prediction += prediction.shape[0]
    
  acc = correct_prediction/total_prediction
  print(f'Accuracy: {acc*100:.2f}, Total items: {total_prediction}')
  return acc, total_prediction
