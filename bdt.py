from functions import MSE, gini_index, weighted_gini_index, print_tree, scale_tree, accuracy_ratio, scan_WP, roc_integral
from tree_base import TreeBase
import random
from math import exp, log
# SimpleTree
# RandomForest



class BoostedDecisionTree(TreeBase):

  def __init__(self):
    pass

  # Override
  # Create a terminal node value
  def to_terminal(self, group):
    outcomes = [row[-1] for row in group]
    if self.boost == 'gradient':
      leaf_value = sum(outcomes)/len(outcomes) # regression
      return leaf_value
    if self.boost == 'adaptive':
      leaf_value = max(set(outcomes), key=outcomes.count) # -1 or 1
      return -1 if leaf_value==0 else leaf_value

  # Override
  def get_split(self, dataset, nCut, weights=None): #nCut is not used
    # original data set has label 1 or 0
    # psudo-residual data set has label a float value between -1 and 1
    # requireing row[-1] > 0 condition, we can perform binary classification
    # with psudo-residual dataset
    if weights==None:
      weights = [1.]*len(dataset)
    class_values = list(set(row[-1]>0 for row in dataset))
       
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    random_var = True
    if random_var == True:
      variables = [random.randrange(len(dataset[0])-1)]
    else:
      variables = range(len(dataset[0])-1)
    for index in variables:
      for row in dataset:
        groups, weight_groups = self.test_split(index, row[index], dataset, weights)
        gini = weighted_gini_index(groups, class_values, weight_groups)
        if gini < b_score:
          #print('updated', gini, b_score, len(groups[0]),len(groups[1]),class_values)
          b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

  # Override
  # Build a decision tree
  def build_tree(self, train, max_depth, min_size, nCut, weights):
    tree = self.get_split(train, nCut, weights)
    self.split(tree, max_depth, min_size, 1, nCut)
    return tree

  def bagging(self, dataset, fraction):
    n_dataset     = len(dataset)
    n_sampling    = int(n_dataset*fraction)
    subsample = random.sample(dataset, n_sampling)
    return subsample
 
  def train(self, max_depth, min_size, nCut=50, n_trees=100, bagging_fraction=0.5, loss_function='MSE', boost_algo='gradient'):

    self.trees = list()
    if boost_algo == 'gradient':
      boost_algo = self.gradient_boost
    elif boost_algo == 'adaptive':
      boost_algo = self.adaptive_boost

    baggedBoost = not bagging_fraction==1
    for i in range(n_trees):
      if baggedBoost == True:
        subsamples = self.bagging(self._reader.dataset_train, bagging_fraction)
      else:
        subsamples = self._reader.dataset_train
      tree_new = boost_algo(subsamples, max_depth, min_size, nCut, loss_function)
      self.trees.append(tree_new)
      print_tree(tree_new)
      print(i, self.get_predictions(self.trees, [self._reader.dataset_train[0]]), self._reader.dataset_train[0][-1])

  def gradient_boost(self, dataset, max_depth, min_size, nCut, loss_function):

    self.boost = 'gradient'
    tree_new = None
    if len(self.trees)==0:
      weights = [1.]*len(dataset)
      tree_new = self.build_tree(dataset, max_depth, min_size, nCut, weights) #initial tree
    else:
      psudo_residual = self.get_psudo_residual(dataset, loss_function)
      weights = [ abs(row[-1]) for row in psudo_residual ]
      tree_new = self.build_tree(psudo_residual, max_depth, min_size, nCut, weights)
    gamma = self.get_gamma(tree_new, dataset, loss_function)
    #print_tree(tree_new)
    scale_tree(tree_new, gamma)
    #print('gamma : %f'%gamma)
    #print_tree(tree_new)
      
    return tree_new

  def get_psudo_residual(self, dataset, loss_function):
    psudo_residual = list()
    if loss_function == 'MSE':
      print(dataset[0][-1] - sum([self.predict(tree, dataset[0]) for tree in self.trees]))
      for row in dataset:
        prediction = sum([ self.predict(tree, row) for tree in self.trees]) # prediction from current model
        row_new = [ val for val in row] # copy row
        row_new[-1] = row[-1] - prediction # modify last column to y-\hat{y}
        psudo_residual.append(row_new)
    elif loss_function == 'binomial_log_likelihood':
      print(dataset[0][-1] - sum([self.predict(tree, dataset[0]) for tree in self.trees]))
      for row in dataset:
        prediction = sum([ self.predict(tree, row) for tree in self.trees]) # prediction from current model
        row_new = [ val for val in row] # copy row
        row_new[-1] = (-2*row[-1]*prediction*exp(-2*row[-1]*prediction))/(1+exp(-2*row[-1]*prediction))
        psudo_residual.append(row_new)
    else:
      raise Exception('unsupported loss function')
    return psudo_residual


  def adaptive_boost(self, dataset, max_depth, min_size, nCut, loss_function): # loss_function is dummy variable

    self.boost = 'adaptive'
    tree_new = None
    if len(self.trees)==0:
      self.adaptive_weights = [1./float(len(dataset))]*len(dataset)
      tree_new = self.build_tree(dataset, max_depth, min_size, nCut, self.adaptive_weights) #initial tree
    else:
      alpha = self.get_adaptive_weights(dataset)
      #print(set([ data[-1] for data in psudo_residual]))
      tree_new = self.build_tree(dataset, max_depth, min_size, nCut, self.adaptive_weights)
      # normalize adaptive_weights
      sum_weights = float(sum(self.adaptive_weights))
      self.adaptive_weights = [ weight/sum_weights for weight in self.adaptive_weights]
      scale_tree(tree_new, alpha)
      
    return tree_new

  def get_adaptive_weights(self,dataset):
    print(dataset[0][-1] - sum([self.predict(tree, dataset[0]) for tree in self.trees]))
    correct, miss = [], []
    for row in dataset:
      prediction = sum([ self.predict(tree, row) for tree in self.trees]) # prediction from current model
      label = row[-1]
      if row[-1] == prediction:
        correct.append(1)
        miss.append(0)
      else:
        correct.append(0)
        miss.append(1)
    n_correct, n_miss = sum(correct), sum(miss)
    r_miss = n_miss/(n_correct + n_miss)
      
    alpha = 0.
    if not r_miss == 0:
      alpha  = log((1-r_miss)/r_miss)
    else:
      alpha = 0
    for i in range(len(dataset)):
      self.adaptive_weights[i] *= exp(alpha) if miss[i] == 1 else 1.
    return alpha


  #retrieve optimal value of gamma such that minimize loss function
  def get_gamma(self, tree_new, dataset, loss_function):
    gamma = 0
    if loss_function == 'MSE':
      #sum(data - predict_old_trees)/sum(predict_new_tree)
      predictions_trees = self.get_predictions(self.trees, dataset)
      predictions_tree_new = self.get_predictions(tree_new, dataset)

      #optimal value of gamma
      for i, row in enumerate(dataset):
        gamma += (row[-1] - sum(predictions_trees[i]))*predictions_tree_new[i]
      gamma /= sum([ prediction**2  for prediction in predictions_tree_new])
      
    elif loss_function == 'binomial_log_likelihood':
      gamma = 1.
    else:
      raise Exception('unsupported loss function')
    return gamma
      

  def get_predictions(self, trees, dataset):
    predictions = list()
    if type(trees) == dict: #recognize as single tree
      for row in dataset:
        prediction = self.predict(trees, row)
        predictions.append(prediction)
    elif type(trees) == list: # recognize as list of trees
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
    prediction_sum_train = [ sum(prediction) for prediction in self.predictions_train]
    prediction_sum_test =  [ sum(prediction) for prediction in self.predictions_test]
    actual_train = [row[-1] for row in self._reader.dataset_train]
    actual_test  = [row[-1] for row in self._reader.dataset_test ]
    accuracy_train = accuracy_ratio(actual_train, prediction_sum_train, 0.5)
    accuracy_test  = accuracy_ratio(actual_test ,  prediction_sum_test, 0.5)
    #auc_train = roc_integral(actual_train, prediction_sum_train)
    #auc_test = roc_integral(actual_test, prediction_sum_test)
    print("Accuracy train : %f"%accuracy_train)
    print("Accuracy test  : %f"%accuracy_test)
    #print("AUC train : %f"%auc_train)
    #print("AUC test  : %f"%auc_test)

