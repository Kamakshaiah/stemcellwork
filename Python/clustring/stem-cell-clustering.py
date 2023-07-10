# https://builtin.com/data-science/data-clustering-python

##import os
##print(os.getcwd()) # D:\Research\papers\HC\Stem_Cells\stemcell-tm\Python

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

file = pd.read_csv('')
##print(df.head())

from sklearn.cluster import KMeans

##X = df[['Age', 'Spending Score (1-100)']].copy()

abstracts = file['Abstract'].copy()

vec = CountVectorizer()
X = vec.fit_transform(abstracts)
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

# stem 
stem_vecs = df.filter(like='stem').columns
a = (stem_vecs.to_numpy() == 0).mean() # 0.9604485396383866
stem_vecs_dense = stem_vecs.query('controversystem >0 | ecosystem >0 | epistemic > 0 | eurostemcell > 0 | meristematic > 0 | metabostemness > 0 | metabostemnessspecific > 0 | microsystems > 0 | stem > 0 | stemness > 0 | stems > 0 | system > 0 | systematic > 0 | systematically > 0 | systemic > 0 | systems > 0')
a = (stem_vecs_dense.to_numpy() == 0).mean() # 0.9212257617728532

# cell 
cell_vecs = df.filter(like='cell')
a = (cell_vecs.to_numpy() ==0).mean() # 0.9585282589455051

cell_vecs_dense = cell_vecs.query('acellular > 0 | agreementcell > 0 | articell > 0 | cell > 0 | cellbased > 0 | celle > 0 | cellforcure > 0 | cells > 0 | cellulaire > 0 | cellular > 0 | eurostemcell > 0 | excellence > 0 | excellent > 0 | extracellular > 0 | icell > 0 | immunocellular > 0 | intracellular > 0 | multicellular > 0 | pointscell > 0 | varicella > 0 | xcelligence > 0 | xenocells > 0 ')
a = (cell_vecs_dense.to_numpy() ==0).mean() # 0.929004329004329

# sell 
sell_vec = df.filter(regex='sell')
a = (sell_vec.to_numpy() ==0).mean() # sparcity 0.9954798331015299
print(a)


sell_vec_dense = sell_vec.query('sell > 0 | sellers > 0 | selling > 0 | sells > 0')
a = (sell_vec_dense.to_numpy() ==0).mean() # sparcity 0.7045454545454546 (reduced)

# marketing 
marketing_vecs = df.filter(like='marketing')
a = (marketing_vecs.to_numpy() ==0).mean() # sparcity 0.8567454798331016

marketing_vecs_dense = marketing_vecs.query('marketing > 0 | postmarketing > 0 | premarketing > 0')
a = (marketing_vecs_dense.to_numpy() ==0).mean() # sparcity 0.6622950819672131

# customer 
customer_vecs = df.filter(like='customer')
a = (customer_vecs.to_numpy() ==0).mean() # 0.9965229485396384

customer_vecs_dense = customer_vecs.query('customer > 0 | customers > 0 ')
a = (customer_vecs_dense.to_numpy() ==0).mean() # 0.5

# affordability 
affordability_vecs = df.filter(like='affordability')
affordability_vecs.query('affordability > 0')

affordability_vecs_dense = affordability_vecs.query('affordability > 0')
                   
a = (affordability_vecs.to_numpy() ==0).mean() # 0.9986091794158554
a = (affordability_vecs_dense.to_numpy() ==0).mean() # 0.0
                   
# familiarity
# no records

# inclination
# no reords

# therapy

therapy_vecs = df.filter(like='therapy')
a = (therapy_vecs.to_numpy() ==0).mean() # 0.9483409497317703
a = (therapy_vecs_dense.to_numpy() ==0).mean() # 0.8311688311688312

# patient care
patient_vecs = df.filter(like='patient')
a = (patient_vecs.to_numpy() ==0).mean() #0.7899860917941586
patient_vecs_dense = patient_vecs[patient_vecs.to_numpy() > 0]
a = (patient_vecs_dense.to_numpy() ==0).mean() # 0.31456953642384106

# pregnancy

pregnancy_vecs = df.filter(like='pregnancy')
a = (pregnancy_vecs.to_numpy() > 0).mean() # 0.004172461752433936

pregnancy_vecs[pregnancy_vecs.to_numpy() > 0]
pregnancy_vecs_dense = pregnancy_vecs[pregnancy_vecs.to_numpy() > 0]
a = (pregnancy_vecs_dense.to_numpy()).mean() # 1.0
a = (pregnancy_vecs_dense.to_numpy() > 0).mean()
pregnancy_vecs.to_csv('pregnancy.csv')

# delivery

delivery_vecs = df.filter(like='delivery')
delivery_vecs.to_csv('delivery.csv')
a = (delivery_vecs.to_numpy() == 0).mean() # 0.972183588317107
a = (delivery_vecs_dense.to_numpy() > 0).mean() # 1.0

# puerperal
# no docs


## clustering 

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

