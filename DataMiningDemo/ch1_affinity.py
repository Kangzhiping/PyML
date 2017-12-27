# 亲和性分析，生成数据并计算支持度，置信度。统计结果
# coding: utf-8
import numpy as np

X = np.zeros((100, 5), dtype='bool')
#X = np.zeros((100, 5))
features = ["bread", "milk", "cheese", "apples", "bananas"]

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        X[i][j] = np.random.randint(2)
    if X[i].sum() == 0:
        X[i][X.shape[1] -1] = True
print(X[:5])
"""
X = np.random.randint(2, size = (X.shape[0], X.shape[1]))
print(X[:5])
"""
"""    if np.random.random() < 0.3:
        if i == 1: print(np.random.random())
        # A bread winner
        X[i][0] = 1
        if np.random.random() < 0.5:
            # Who likes milk
            X[i][1] = 1
        if np.random.random() < 0.2:
            # Who likes cheese
            X[i][2] = 1
        if np.random.random() < 0.25:
            # Who likes apples
            X[i][3] = 1
        if np.random.random() < 0.5:
            # Who likes bananas
            X[i][4] = 1
    else:
        # Not a bread winner
        if np.random.random() < 0.5:
            if i == 1: print(np.random.random())
            # Who likes milk
            X[i][1] = 1
            if np.random.random() < 0.2:
                # Who likes cheese
                X[i][2] = 1
            if np.random.random() < 0.25:
                # Who likes apples
                X[i][3] = 1
            if np.random.random() < 0.5:
                # Who likes bananas
                X[i][4] = 1
        else:
            if np.random.random() < 0.8:
                # Who likes cheese
                X[i][2] = 1
            if np.random.random() < 0.6:
                # Who likes apples
                X[i][3] = 1
            if np.random.random() < 0.7:
                # Who likes bananas
                X[i][4] = 1
    if X[i].sum() == 0:
        X[i][4] = 1  # Must buy something, so gets bananas
"""

print(X[:5])

np.savetxt("affinity_dataset01.txt", X, fmt='%d')

# 分析数据


# coding: utf-8

import numpy as np
dataset_filename = "affinity_dataset01.txt"
X = np.loadtxt(dataset_filename)
n_samples, n_features = X.shape
print("This dataset has {0} samples and {1} features".format(n_samples, n_features))


# In[3]:

print(X[:5])


# In[4]:

# The names of the features, for your reference.
features = ["bread", "milk", "cheese", "apples", "bananas"]


# In our first example, we will compute the Support and Confidence of the rule "If a person buys Apples, they also buy Bananas".

# In[5]:

# First, how many rows contain our premise: that a person is buying apples
num_apple_purchases = 0
for sample in X:
    if sample[3] == 1:  # This person bought Apples
        num_apple_purchases += 1
print("{0} people bought Apples".format(num_apple_purchases))


# In[6]:

# How many of the cases that a person bought Apples involved the people purchasing Bananas too?
# Record both cases where the rule is valid and is invalid.
rule_valid = 0
rule_invalid = 0
for sample in X:
    if sample[3] == 1:  # This person bought Apples
        if sample[4] == 1:
            # This person bought both Apples and Bananas
            rule_valid += 1
        else:
            # This person bought Apples, but not Bananas
            rule_invalid += 1
print("{0} cases of the rule being valid were discovered".format(rule_valid))
print("{0} cases of the rule being invalid were discovered".format(rule_invalid))


# In[20]:

# Now we have all the information needed to compute Support and Confidence
support = rule_valid  # The Support is the number of times the rule is discovered.
confidence = rule_valid / num_apple_purchases
print("The support is {0} and the confidence is {1:.3f}.".format(support, confidence))
# other print method
print("The support is %d and the confidence is %.3f" % (support, confidence))
# Confidence can be thought of as a percentage using the following:
print("As a percentage, that is {0:.1f}%.".format(100 * confidence))



from collections import defaultdict
# Now compute for all possible rules
valid_rules = defaultdict(int)
invalid_rules = defaultdict(int)
num_occurences = defaultdict(int)

for sample in X:
    for premise in range(n_features):
        if sample[premise] == 0: continue
        # Record that the premise was bought in another transaction
        num_occurences[premise] += 1
        for conclusion in range(n_features):
            if premise == conclusion:  # It makes little sense to measure if X -> X.
                continue
            if sample[conclusion] == 1:
                # This person also bought the conclusion item
                valid_rules[(premise, conclusion)] += 1
            else:
                # This person bought the premise, but not the conclusion
                invalid_rules[(premise, conclusion)] += 1
support = valid_rules
confidence = defaultdict(float)
for premise, conclusion in valid_rules.keys():
    confidence[(premise, conclusion)] = valid_rules[(premise, conclusion)] / num_occurences[premise]
    print(confidence[(premise, conclusion)])


# In[24]:

for premise, conclusion in confidence:
    premise_name = features[premise]
    conclusion_name = features[conclusion]
    print("Rule: If a person buys {0} they will also buy {1}".format(premise_name, conclusion_name))
    print(" - Confidence: {0:.3f}".format(confidence[(premise, conclusion)]))
    print(" - Support: {0}".format(support[(premise, conclusion)]))
    print("")


# In[25]:

def print_rule(premise, conclusion, support, confidence, features):
    premise_name = features[premise]
    conclusion_name = features[conclusion]
    print("Rule: If a person buys {0} they will also buy {1}".format(premise_name, conclusion_name))
    print(" - Confidence: {0:.3f}".format(confidence[(premise, conclusion)]))
    print(" - Support: {0}".format(support[(premise, conclusion)]))
    print("")


premise = 1
conclusion = 3
print_rule(premise, conclusion, support, confidence, features)


# Sort by support
from pprint import pprint
pprint(list(support.items()))


from operator import itemgetter
sorted_support = sorted(support.items(), key=itemgetter(1), reverse=True)

for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_support[index][0]
    print_rule(premise, conclusion, support, confidence, features)


sorted_confidence = sorted(confidence.items(), key=itemgetter(1), reverse=True)

for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    print_rule(premise, conclusion, support, confidence, features)




