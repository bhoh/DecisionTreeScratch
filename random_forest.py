from functions import MSE, gini_index, weighted_gini_index, print_tree, scale_tree, accuracy_ratio, scan_WP, roc_integral
from tree_base import TreeBase
import random
from math import exp, log
# SimpleTree
# RandomForest




class RandomForest(TreeBase):

  def __init__(self):
    pass

  def bagging(self, dataset, fraction):
    n_dataset     = len(dataset)
    n_sampling    = int(n_dataset*fraction)
    subsample = random.sample(dataset, n_sampling)
    return subsample
    

  def train(self, max_depth, min_size, nCut=50, n_trees=20, bagging_fraction=0.5):
    self.trees = list()
    for i in range(n_trees):
      subsamples = self.bagging(self._reader.dataset_train, bagging_fraction)
      tree = self.build_tree(subsamples, max_depth, min_size, nCut)
      #print_tree(tree)
      self.trees.append(tree)

  def get_predictions(self, trees, dataset):
    predictions = list()
    for row in dataset:
      prediction = list()
      for tree in trees:
        prediction.append(self.predict(tree, row))
      predictions.append(prediction)
    return predictions

  def predict(self, tree, row):
    if row[tree['index']] < tree['value']:
      if isinstance(tree['left'], dict):
        return self.predict(tree['left'], row)
      else:
        return tree['left']
    else:
      if isinstance(tree['right'], dict):
        return self.predict(tree['right'], row)
      else:
        return tree['right']

  def eval(self):
    self.predictions_train = self.get_predictions(self.trees, self._reader.dataset_train)
    self.predictions_test  = self.get_predictions(self.trees, self._reader.dataset_test)
    #print("Predictions train")
    #print(self.predictions_train)
    #print("Predictions test")
    #print(self.predictions_test)
    prediction_avg_train = [ sum(prediction)/len(prediction ) for prediction in self.predictions_train]
    prediction_avg_test =  [ sum(prediction)/len(prediction) for prediction in self.predictions_test]
    accuracy_train = accuracy_ratio([row[-1] for row in self._reader.dataset_train], prediction_avg_train, 0.5)
    accuracy_test  = accuracy_ratio([row[-1] for row in self._reader.dataset_test],  prediction_avg_test, 0.5)
    print("Accuracy train : %f"%accuracy_train)
    print("Accuracy test  : %f"%accuracy_test)



