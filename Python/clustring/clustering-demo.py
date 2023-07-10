##import os
##print(os.getcwd()) # D:\Research\papers\HC\Stem_Cells\stemcell-tm\Python

import pandas as pd

df = pd.read_csv('Mall_Customers.csv')
##print(df.head())

from sklearn.cluster import KMeans

X = df[['Age', 'Spending Score (1-100)']].copy()

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

##plt.plot(range(1, 11), wcss)
##plt.title('Selecting the Numbeer of Clusters using the Elbow Method')
##
##plt.xlabel('Clusters')
##plt.ylabel('WCSS')
##plt.show()

kmeans = KMeans(n_clusters=4, random_state=0)
kmeans_fit = kmeans.fit(X)

##sns.scatterplot(data=df, x="Age", y="Spending Score (1-100)", hue=kmeans_fit.cluster_centers_)

colors = ['red', 'blue', 'purple', 'green', 'yellow']
ax = sns.scatterplot(X[:, 0], X[:, 1], hue=kmeans.labels_, palette=colors, alpha=0.5, s=7)
ax = sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], hue=range(num_clusters), palette=colors, s=20, ec='black', legend=False, ax=ax)
plt.show()

# guassian mixture model

from sklearn.mixture import GaussianMixture

n_clusters = 3
gmm_model = GaussianMixture(n_components=n_clusters)
gmm_model.fit(X)

cluster_labels = gmm_model.predict(X)
X = pd.DataFrame(X)
X['cluster'] = cluster_labels

for k in range(0,n_clusters):
    data = X[X["cluster"]==k]
    plt.scatter(data["Age"],data["Spending Score (1-100)"])

plt.title("Clusters Identified by Guassian Mixture Model")    
plt.ylabel("Spending Score (1-100)")
plt.xlabel("Age")
plt.show()

