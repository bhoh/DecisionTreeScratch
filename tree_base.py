from functions import MSE, gini_index, weighted_gini_index, print_tree, scale_tree, accuracy_ratio, scan_WP, roc_integral
import random
from math import exp, log
# SimpleTree
# RandomForest


class TreeBase():

  def __init__(self):
    pass

  def set_reader(self, reader):
    self._reader = reader

  # Split a dataset based on an attribute and an attribute value
  def test_split(self, index, value, dataset, weights=None):
    if weights==None:
      weights = [1.]*len(dataset)
    left, right = list(), list()
    weight_left, weight_right = list(), list()
    for i, row in enumerate(dataset):
      if row[index] < value:
        left.append(row)
        weight_left.append(weights[i])
      else:
        right.append(row)
        weight_right.append(weights[i])
    return (left, right), (weight_left, weight_right)

  # Select the best split point for a dataset
  #def get_split(self, dataset, nCut):
  #  class_values = list(set(row[-1] for row in dataset))
  #  b_index, b_value, b_score, b_groups = 999, 999, 999, None
  #  for index in range(len(dataset[0])-1):
  #    row_min = min([row[index] for row in dataset])
  #    row_max = max([row[index] for row in dataset])
  #    step = (row_max-row_min)/nCut
  #    print(row_min, row_max)
  #    for iCut in range(nCut+1):
  #      value = row_min + iCut*step
  #      print(value)
  #      groups = self.test_split(index, value, dataset)
  #      gini = gini_index(groups, class_values)
  #      if gini < b_score:
  #        b_index, b_value, b_score, b_groups = index, value, gini, groups
  #  return {'index':b_index, 'value':b_value, 'groups':b_groups}

  def get_split(self, dataset, nCut): #nCut is not used
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    random_var = True
    if random_var == True:
      variables = [random.randrange(len(dataset[0])-1)]
    else:
      variables = range(len(dataset[0])-1)
    for index in variables:
      for row in dataset:
        groups, _ = self.test_split(index, row[index], dataset)
        gini = gini_index(groups, class_values)
        if gini < b_score:
          #print('updated', gini, b_score, len(groups[0]),len(groups[1]),class_values)
          b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

  # Create a terminal node value
  def to_terminal(self, group):
    outcomes = [row[-1] for row in group]
    #return max(set(outcomes), key=outcomes.count) #CART
    return sum(outcomes)/len(outcomes) # regression

  # Create child splits for a node or make terminal
  def split(self, node, max_depth, min_size, depth, nCut):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
      node['left'] = node['right'] = self.to_terminal(left + right)
      return
    # check for max depth
    if depth >= max_depth:
      node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
      return
    # process left child
    if len(left) <= min_size:
      node['left'] = self.to_terminal(left)
    else:
      node['left'] = self.get_split(left, nCut)
      self.split(node['left'], max_depth, min_size, depth+1, nCut)
    # process right child
    if len(right) <= min_size:
      node['right'] = self.to_terminal(right)
    else:
      node['right'] = self.get_split(right, nCut)
      self.split(node['right'], max_depth, min_size, depth+1, nCut)

  # Build a decision tree
  def build_tree(self, train, max_depth, min_size, nCut):
    tree = self.get_split(train, nCut)
    self.split(tree, max_depth, min_size, 1, nCut)
    return tree


