def median(a_list):
  n = len(a_list)
  s = sorted(a_list)
  return (sum(s[n//2-1:n//2+1])/2.0, s[n//2])[n % 2] if n else None


def gini_index(groups, classes):
  # count all samples at split point
  n_instances = float(sum([len(group) for group in groups]))
  # sum weighted Gini index for each group
  gini = 0.0
  for group in groups:
    size = float(len(group))
    # avoid divide by zero
    if size == 0:
      continue
    score = 0.0
    # score the group based on the score for each class
    for class_val in classes:
      p = [row[-1] for row in group].count(class_val) / size
      score += p * p
    # weight the group score by its relative size
    gini += (1.0 - score) * (size / n_instances)
  return gini

def MSE(index, groups):
  MSE = 0.0
  for group in groups:
    size = float(len(group))
    # avoid divide by zero
    if size == 0:
      continue

    group_mean = float(sum([ row[index] for row in group ]))/size
    group_MSE = sum([ (1/2)*(row[-1]-group_mean)**2 for row in group ])
    MSE += group_MSE
  return MSE


def weighted_gini_index(groups, classes, weight_groups=None):
  #init weight_group
  if weight_groups == None:
    weight_groups = [None]*len(groups)
    for i, group in enumerate(groups):
      weight_groups[i] = [1.]*len(group)
  #estimate sum of weights
  sum_weight_groups = float(sum([ float(sum(weight_group)) for weight_group in weight_groups ]))
  # count all samples at split point
  # sum weighted Gini index for each group
  gini = 0.0
  for i, group in enumerate(groups):
    sum_weight_group = float(sum(weight_groups[i]))
    # avoid divide by zero
    if sum_weight_group == 0:
      continue
    score = 0.0
    # score the group based on the score for each class
    for class_val in classes:
      count_class_val = 0.
      # count class value
      for j, row in enumerate(group):
        if (row[-1]>0) == class_val:
          count_class_val += weight_groups[i][j]
      p = count_class_val / sum_weight_group
      #print(class_val,p)
      score += p * p
    # weight the group score by its relative sum_weight_group
    # and weight by group for adaptive boost
    gini += (1.0 - score) * (sum_weight_group / sum_weight_groups)
  return gini



# Print a decision tree
def print_tree(node, depth=0):
  if isinstance(node, dict):
    print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
    print_tree(node['left'], depth+1)
    print_tree(node['right'], depth+1)
  else:
    print('%s[%s]' % ((depth*' ', node)))


def scale_tree(node, gamma):
  if isinstance(node, dict):
    if isinstance(node['left'], dict):
      scale_tree(node['left'], gamma)
    else:
      node['left'] = node['left']*gamma
    if isinstance(node['right'], dict):
      scale_tree(node['right'], gamma)
    else:
      node['right'] = node['right']*gamma
  else:
    pass

# Calculate accuracy ratio
# accuracy   : (total pass and correct) / (total correct)
def accuracy_ratio(actual, predicted):
  correct = 0
  for i in range(len(actual)):
    if actual[i] == predicted[i]:
      correct += 1
  return correct / float(len(actual))

def accuracy_ratio(actual, predicted, WP):
  correct = 0
  for i in range(len(actual)):
    if actual[i] == (predicted[i] > WP):
      correct += 1
  return correct / float(len(actual))

# TP : True positive,  classified true and actually true
# FP : False positive, classified true and actually false
# TN : True negative
# FN : False negative
# TPR : TP/(TP+FN)  : sensitivity in statistics, signal efficiency in HEP
# FPR : FP/(FP+TN)  : 1-specificity in statistics, background efficiency in HEP


def confusion_matrix(actual, predicted, WP): 
  TP, FP, TN, FN = 0., 0., 0., 0.
  for i in range(len(predicted)):
    if actual[i] == 1:
      if predicted[i] > WP:
        TP += 1.
      else:
        TN += 1.
    else:
      if predicted[i] > WP:
        FP += 1.
      else:
        FN += 1.
  TPR = TP / (TP + FN)
  FPR = FP / (FP + TN)
  return TPR, FPR

def roc_integral(actual, predicted):
  area = 0.
  TPR, FPR = confusion_matrix(actual, predicted, 1)
  #print(TPR,FPR)
  for step in range(1,1001):
    WP = 1-float(step)/1000.
    TPR_new, FPR_new = confusion_matrix(actual, predicted, WP)
    # FPR*d(TPR) = FPR * (d(TPR)/d(WP)) * d(WP)
    #d(TPR) = TPR_new - TPR
    area += ((TPR+TPR_new)/2) * (FPR_new-FPR)
    #update
    TPR, FPR = TPR_new, FPR_new

  return area


# 1) pass WP,       not pass WP
# 2) true false,    true, false

# efficiency : (total pass) / (total pass and not pass)
# purity     :  (total pass and correct) / (total pass)

def scan_WP(actual, predicted):
  WP = 0.
  step = 0.001
  best_metric = -999; best_WP = -999
  for i in range(int(1/step)):
    correct_ratio, incorrect_ratio = accuracy_ratio(actual, predicted, WP)
    metric = correct_ratio*(correct_ratio + incorrect_ratio) # efficiency \times purity
    if metric > best_metric:
      best_metric = metric
      best_WP = WP
    WP += step
  return best_metric, best_WP

if __name__ == '__main__':

  # gini_index function test
  group1 = [[1],[1],[1],[1]]
  group2 = [[0],[0],[0],[1]]
  groups = (group1,group2)
  classes = [0,1]
  weight1 = [1.,1.,1.,1.]
  weight2 = [1.,1.,1.,3.]
  weights = [weight1, weight2]
  gini = gini_index(groups, classes)
  weighted_gini = weighted_gini_index(groups, classes, weights)
  print(gini, weighted_gini)

