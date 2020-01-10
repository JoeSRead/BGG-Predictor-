from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def tsne_plot(data, label=False,limit=-1, skip=1, figsze=(15,10), **kwargs):
    """TSNE plotter, should be able to take any TSNE arguments.
    label = True will label the resultant scatter plot
    If label is true then the limit argument can be changed to restrict 
    the labeling to a particular subset, in addition skip may be used to skip labels
    """
    tsne = TSNE(**kwargs)
    embed = tsne.fit_transform(data)
    X = [x[0] for x in embed]
    Y = [x[1] for x in embed]
    
    plt.figure(figsize = figsze)
    plt.scatter(X, Y)
    
    if label == True:
        names = [x for x in data['details.name']]
        
        for i,name in enumerate(names[:limit:skip]):
            plt.text(X[i], Y[i], name, fontsize=9)
    plt.axis('off')
    plt.show()