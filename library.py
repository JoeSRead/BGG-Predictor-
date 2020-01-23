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
    def __init__(self, df, perp = 50, n_iter = 1000, init = 'pca', num_from = 'minplayers'):
        """Num_from is the column name that we slice on to get only numeric data
        """
        self.df = df
        self.perp = perp
        self.n_iter = n_iter
        self.init = init
        
        self.df_num = self.df.loc[:, num_from:]
        self.names = self.df['details.name']
        
        self.tsne = TSNE(perplexity = self.perp, n_iter = self.n_iter, init = self.init, random_state = 137)
        self.embed = self.tsne.fit_transform(self.df_num)
        
    def plot(self, fig_size = None, label_every = 0, clusters = False):
        """ The label_every is used to determine what fraction of points should be labeled, if it is not 0 the structure is: names[::label_every] 
        """
        plt.figure(figsize = fig_size)
        
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
            
    
    def graph_data(self, cluster_dist = 3):
        """ Creates the df_labels dataframe that contains the name of each clustered game, its x,y position, its cluster label, and the labels of any clusters within cluster_dist.
        Also creates the df_connected_labels that only contains the connections between clusters of games
        """
        self.df_labels = pd.DataFrame(data={'name': self.names, 'label': self.labels + 2, 'x': self.embed[:].T[0], 'y': self.embed[:].T[1], 'local_labels':[[] for i in range(len(self.embed))]})
        self.df_labels = self.df_labels.loc[self.df_labels['label'] != 1]
        
        for label in set(self.df_labels['label']):
            for index, row in self.df_labels.loc[self.df_labels['label'] == label].iterrows():
                x = self.df_labels.loc[index, 'x']
                y = self.df_labels.loc[index, 'y']
                for index_2, row_2 in self.df_labels.loc[self.df_labels['label'] != label].iterrows():
                    other_label = self.df_labels.loc[self.df_labels['label'] != label].loc[index_2, 'label']
                    x_other = self.df_labels.loc[index_2, 'x']
                    y_other = self.df_labels.loc[index_2, 'y']
                    if (x - x_other)**2 + (y - y_other)**2 <= cluster_dist**2:
                        self.df_labels.loc[index, 'local_labels'].append(other_label)
        
        self.connected_labels_lst = list(set(self.df_labels[self.df_labels['local_labels'].map(lambda d: len(d)) > 0]['label']))
        self.df_connected_labels = self.df_labels[self.df_labels['local_labels'].map(lambda d: len(d)) > 0]
        
        self.label_edges = []

        for lbl in range(2, self.labels.max() + 3):
            local_lst = []
            for index, row in self.df_connected_labels.loc[self.df_connected_labels['label'] == lbl].iterrows():
                local_lst.append(row['local_labels'])
            local_lst = [item for sublist in local_lst for item in sublist]
            for cnct in set(local_lst):
                self.label_edges.append((lbl, cnct, {'weight' : round(local_lst.count(cnct)/len(local_lst),3)}))
        
        
        
    def draw_graph(self, all_nodes = False):
        """Only the all_nodes = False graph is nameble, this is done by manually changing the connected labels list
        """
        
        if all_nodes:
            self.G = nx.Graph()
            self.G.add_nodes_from(range(2, self.labels.max() + 3))
            self.G.add_edges_from(self.label_edges)
            
            elarge = [(u, v) for (u, v, d) in self.G.edges(data=True) if d["weight"] > 0.667]
            emed = [(u, v) for (u, v, d) in self.G.edges(data=True) if (d["weight"] > 0.334 and d['weight'] <= 0.667)]
            esmall = [(u, v) for (u, v, d) in self.G.edges(data=True) if d["weight"] <= 0.334]
            
            pos = nx.spring_layout(self.G)
            
            nx.draw_networkx_nodes(self.G, pos, node_size=700)

            nx.draw_networkx_edges(self.G, pos, edgelist = elarge, width = 12, edge_color='red')
            nx.draw_networkx_edges(self.G, pos, edgelist = emed, width = 8, edge_color='red')
            nx.draw_networkx_edges(self.G, pos, edgelist = esmall, width = 4, edge_color='red')

            nx.draw_networkx_labels(self.G, pos)

            plt.axis("off")
            plt.show();
        
        
        else:
            self.G = nx.Graph()
            self.G.add_nodes_from(self.connected_labels_lst)
            self.G.add_edges_from(self.label_edges)
                      
            elarge = [(u, v) for (u, v, d) in self.G.edges(data=True) if d["weight"] > 0.667]
            emed = [(u, v) for (u, v, d) in self.G.edges(data=True) if (d["weight"] > 0.334 and d['weight'] <= 0.667)]
            esmall = [(u, v) for (u, v, d) in self.G.edges(data=True) if d["weight"] <= 0.334]
            
            pos = nx.spring_layout(self.G)
            
            nx.draw_networkx_nodes(self.G, pos, node_size=700)

            nx.draw_networkx_edges(self.G, pos, edgelist = elarge, width = 12, edge_color='red')
            nx.draw_networkx_edges(self.G, pos, edgelist = emed, width = 8, edge_color='red')
            nx.draw_networkx_edges(self.G, pos, edgelist = esmall, width = 4, edge_color='red')

            nx.draw_networkx_labels(self.G, pos)

            plt.axis("off")
            plt.show();
