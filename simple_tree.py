from functions import MSE, gini_index, weighted_gini_index, print_tree, scale_tree, accuracy_ratio, scan_WP, roc_integral
from tree_base import TreeBase
import random
from math import exp, log
# SimpleTree
# RandomForest



class SimpleTree(TreeBase):

  def __init__(self):
    pass

  def train(self, max_depth, min_size, nCut=50):
    self.tree = self.build_tree(self._reader.dataset_train, max_depth, min_size, nCut)
    #print_tree(self.tree)

  def get_predictions(self, tree, dataset):
    predictions = list()
    for row in dataset:
      prediction = self.predict(tree, row)
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
    self.predictions_train = self.get_predictions(self.tree, self._reader.dataset_train)
    self.predictions_test  = self.get_predictions(self.tree, self._reader.dataset_test)
    print("Predictions train")
    print(self.predictions_train)
    print("Predictions test")
    print(self.predictions_test)
    accuracy_train = accuracy_ratio([row[-1] for row in self._reader.dataset_train], self.predictions_train, 0.5)
    accuracy_test  = accuracy_ratio([row[-1] for row in self._reader.dataset_test], self.predictions_test, 0.5)
    print("Accuracy train : %f"%accuracy_train)
    print("Accuracy test  : %f"%accuracy_test)




