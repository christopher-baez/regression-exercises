import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

import env
import wrangle


def plot_variable_pairs(df):
    """
    this function takes in a dataframe and prints out a pair plot
    by only taking 10000 row sample for efficiency.

    parameters
        any dataframe
    return
        a pair plot of the variables
    """
    sns.pairplot(df.sample(10000), corner=True, kind='reg')


plt.show()


def plot_categorical_and_continuous_vars(train):
    """
    this function takes in a dataframe and displays different visuals to help
    visualize correlations between different variables in the zillow df

    parameters
        train df
    output
        visuals
    """
    # creating 2 variable continuous and categorical
    train_cat = ['county']
    train_cont = train[['bedrooms', 'bathrooms', 'area', 'appraisal', 'tax', 'yearbuilt']]

    # creating a line plot
    sns.lineplot(train.sample(10000), x='yearbuilt', y='appraisal', hue='county')
    plt.ylim(0, 2000000)
    plt.show()

    corrs = train_cont.corr(method='spearman')
    # seeing the correlation between each continuous variable
    sns.heatmap(corrs, cmap='PRGn', annot=True)
    plt.show()

    ttrain = train.drop('appraisal', axis=1)

    for var in ttrain.columns:
        sns.barplot(train.sample(10000), x=var, y='appraisal')
        plt.show()




