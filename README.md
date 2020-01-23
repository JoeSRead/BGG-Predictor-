# BGG-Predictor-

Technologies:
* Python
* Pandas
* Matplotlib
* Seaborn
* SQL
* Sklearn
* t-SNE
* Custering
* HDBSCAN
* NetworkX

Perplexity is how many of its NN (in higher dim) a point is attracted to once it's in 2D [1]

Possible pipeline: 
(i) downsample a large data set down to some manageable size; 
(ii) run t-SNE using large perplexity to
preserve global geometry; 
(iii) position all the remaining points on the resulting t-SNE plot using k
nearest neighbours; 
(iv) use it as initialisation to run t-SNE on the whole data set [1]


[1]: Kobak, Dmitry, and Philipp Berens. "The art of using t-SNE for single-cell transcriptomics." Nature communications 10.1 (2019): 1-14.