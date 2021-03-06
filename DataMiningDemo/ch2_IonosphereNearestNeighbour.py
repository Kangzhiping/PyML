# K 近邻算法做预测分类，学习使用sklearn, 训练，预测，流水线
# 使用UCI 机器学习库里面的‘电离层’数据集

import os
home_folder = os.path.expanduser("./")
print(home_folder)
data_folder = os.path.join(home_folder, "ch2_data")
data_filename = os.path.join(data_folder, "ionosphere.ch2_data")
print(data_filename)

import csv
import numpy as np

# Size taken from the dataset and is known
X = np.zeros((351, 34), dtype='float')
y = np.zeros((351,), dtype='bool')

with open(data_filename, 'r') as input_file:
    reader = csv.reader(input_file)
    for i, row in enumerate(reader):
        # Get the ch2_data, converting each item to a float
        data = [float(datum) for datum in row[:-1]]
        # Set the appropriate row in our dataset
        X[i] = data
        # 1 if the class is 'g', 0 otherwise
        y[i] = row[-1] == 'g'

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=14)
print("There are {} samples in the training dataset".format(X_train.shape[0]))
print("There are {} samples in the testing dataset".format(X_test.shape[0]))
print("Each sample has {} features".format(X_train.shape[1]))

from sklearn.neighbors import KNeighborsClassifier

estimator = KNeighborsClassifier()
estimator.fit(X_train, y_train)

y_predicted = estimator.predict(X_test)
accuracy = np.mean(y_test == y_predicted) * 100
print("The accuracy is {0:.1f}%".format(accuracy))

from sklearn.cross_validation import cross_val_score

scores = cross_val_score(estimator, X, y, scoring='accuracy')
average_accuracy = np.mean(scores) * 100
print("The average accuracy is {0:.1f}%".format(average_accuracy))

avg_scores = []
all_scores = []
parameter_values = list(range(1, 21))  # Including 20
for n_neighbors in parameter_values:
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(estimator, X, y, scoring='accuracy')
    avg_scores.append(np.mean(scores))
    all_scores.append(scores)

from matplotlib import pyplot as plt
plt.figure(figsize=(32,20))
plt.plot(parameter_values, avg_scores, '-o', linewidth=5, markersize=24)
plt.show()
#plt.axis([0, max(parameter_values), 0, 1.0])

# In[21]:

for parameter, scores in zip(parameter_values, all_scores):
    n_scores = len(scores)
    plt.plot([parameter] * n_scores, scores, '-o')


# In[23]:

plt.plot(parameter_values, all_scores, 'bx')


# In[26]:

from collections import defaultdict
all_scores = defaultdict(list)
parameter_values = list(range(1, 21))  # Including 20
for n_neighbors in parameter_values:
    for i in range(100):
        estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
        scores = cross_val_score(estimator, X, y, scoring='accuracy', cv=10)
        all_scores[n_neighbors].append(scores)
for parameter in parameter_values:
    scores = all_scores[parameter]
    n_scores = len(scores)
    plt.plot([parameter] * n_scores, scores, '-o')


# In[27]:

plt.plot(parameter_values, avg_scores, '-o')


# In[28]:

X_broken = np.array(X)
# 每隔一行，把第二个值除以10 来破坏数据,结果会变很低
X_broken[:, ::2] /= 10
estimator = KNeighborsClassifier()
original_scores = cross_val_score(estimator, X, y, scoring='accuracy')
print("The original accuracy is {0:.1f}%".format(np.mean(original_scores) * 100))
broken_scores = cross_val_score(estimator, X_broken, y, scoring='accuracy')
print("The broken accuracy is {0:.1f}%".format(np.mean(broken_scores) * 100))


# In[29]:

from sklearn.preprocessing import MinMaxScaler


# In[30]:

# 把特征值转化到0~1 之间，最小为0， 最大为1， 即使破坏之后都还有较高正确率
X_transformed = MinMaxScaler().fit_transform(X_broken)
transformed_scores = cross_val_score(estimator, X_transformed, y, scoring='accuracy')
print("The transformed accuracy is {0:.1f}%".format(np.mean(transformed_scores) * 100))


# In[31]:

# 创建流水线来操作
from sklearn.pipeline import Pipeline
scaling_pipeline = Pipeline([('scale', MinMaxScaler()), ('predict', KNeighborsClassifier())])
pipeline_scores = cross_val_score(scaling_pipeline, X_broken, y, scoring='accuracy')
print("The original accuracy is {0:.1f}%".format(np.mean(pipeline_scores) * 100))
