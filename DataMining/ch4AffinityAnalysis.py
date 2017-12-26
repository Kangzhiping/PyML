import os
data_folder = os.path.join(os.path.expanduser("~"), "workspace\Learning-Data-Mining-with-Python\Chapter 4", "Data")
ratings_filename = os.path.join(data_folder, "ratings.csv")

import pandas as pd

all_ratings = pd.read_csv(ratings_filename)
all_ratings["timestamp"] = pd.to_datetime(all_ratings['timestamp'],unit='s')
print(all_ratings[:5])

# As you can see, there are no review for most movies, such as #213
all_ratings[all_ratings["userId"] == 675].sort_values("movieId")


from collections import defaultdict
def find_frequent_itemsets(favorable_reviews_by_users, k_1_itemsets, min_support):
    counts = defaultdict(int)
    for user, reviews in favorable_reviews_by_users.items():
        for itemset in k_1_itemsets:
            if itemset.issubset(reviews):
                for other_reviewed_movie in reviews - itemset:
                    current_superset = itemset | frozenset((other_reviewed_movie,))
                    counts[current_superset] += 1
    return dict([(itemset, frequency) for itemset, frequency in counts.items() if frequency >= min_support])