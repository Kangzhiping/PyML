
import os
import pandas as pd
import numpy as np
data_folder = os.path.join(os.path.expanduser("~"),"workspace\Learning-Data-Mining-with-Python\Chapter 5", "Data")
data_filename = os.path.join(data_folder, "ad.data")

def convert_number(x):
    print("xxx")
    try:
        return float(x)
    except ValueError:
        return np.nan

from collections import defaultdict
converters = defaultdict(convert_number())  #{i: convert_number for i in range(1558)}
converters[1558] = lambda x: 1 if x.strip() == "ad." else 0

ads = pd.read_csv(data_filename, header=None, converters=converters)
#ads = ads.applymap(lambda x: np.nan if isinstance(x, str) and not x == "ad." else x)
#ads[[0, 1, 2]] = ads[[0, 1, 2]].astype(float)

print(ads)
# In[9]:

#ads = ads.astype(float).dropna()
X = ads.drop(1558, axis=1).values
y = ads[1558]
X.shape
y.shape


# In[10]:

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score

clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X, y, scoring='accuracy')
print("The average score is {:.4f}".format(np.mean(scores)))

'''

# In[10]:

from sklearn.decomposition import PCA
pca = PCA(n_components=5)
Xd = pca.fit_transform(X)


# In[11]:

Xd.shape


# In[18]:

np.set_printoptions(precision=3, suppress=True)
pca.explained_variance_ratio_


# In[19]:


pca.components_[0]


# In[20]:

clf = DecisionTreeClassifier(random_state=14)
scores_reduced = cross_val_score(clf, Xd, y, scoring='accuracy')
print("The average score from the reduced dataset is {:.4f}".format(np.mean(scores_reduced)))


# In[37]:

get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
classes = set(y)
colors = ['red', 'green']
for cur_class, color in zip(classes, colors):
    mask = (y == cur_class).values
    plt.scatter(Xd[mask,0], Xd[mask,1], marker='o', color=color, label=int(cur_class))
plt.legend()

plt.show()


# In[ ]:'''



