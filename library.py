import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import seaborn as sns

import sqlite3
import csv
import time
import requests
from bs4 import BeautifulSoup

from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, ward
import networkx as nx
from hdbscan import HDBSCAN


class BGG:
    def __init__(self, df, perp = 50, n_iter = 1000, init = 'pca'):
        self.df = df
        self.perp = perp
        self.n_iter = n_iter
        self.init = init
        
        self.df_num = self.df.loc[:,'minplayers':]
        self.names = self.df['details.name']
        
        self.tsne = TSNE(perplexity = self.perp, n_iter = self.n_iter, init = self.init, random_state = 137)
        self.embed = self.tsne.fit_transform(self.df_num)
        
    def plot(self, fig_size = None, label_every = 0, clusters = False):
        """ The label_every is used to determine what fraction of points should be labeled, if it is not 0 the structure is: names[::label_every] 
        """
        plt.figure(figsize = None)
        
        if label_every > 0:
            for i,name in enumerate(self.names[::label_every]):
                plt.text(self.embed[i].T[0], self.embed[i].T[1], name, fontsize=9)
                
        if clusters:
            palette = sns.color_palette(n_colors=len(self.clusters.labels_))

            cluster_colors = [sns.desaturate(palette[col], sat) if col >= 0 else (0.5, 0.5, 0.5) for col, sat in zip(self.clusters.labels_, self.clusters.probabilities_)]

            plt.scatter(self.embed[:].T[0], self.embed[:].T[1], c=cluster_colors)
        
        if not clusters:
            plt.scatter(self.embed[:].T[0], self.embed[:].T[1])
        
        plt.axis('off')
        plt.show()
        
    def clusterer(self, min_cluster_size = 5, min_samples = None):
        """ Plot = true plots the t-SNE with proper cluster colours, with label_every != 0 the clusters become named
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        
        self.hdbscan = HDBSCAN(min_cluster_size = self.min_cluster_size, min_samples = self.min_samples)
        self.clusters = self.hdbscan.fit(self.embed)
        
        self.labels = self.clusters.labels_
        self.named_labels = list(zip(self.names, self.labels + 2))
        
        self.cluster_dict = {}
        
        for label in range(1, self.labels.max() + 3):
            namelst = []
            for name, cat in self.named_labels:
                if cat == label:
                    namelst.append(name)
            self.cluster_dict[label] = namelst
            
    
    def graph(self,):