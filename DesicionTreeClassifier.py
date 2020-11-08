from reader import DataReader
from bdt import *
from simple_tree import *
from random_forest import *
# a decision tree class
# with this class we can train a tree
# and wrap multiple trained tree for evaluating a score

class DecisionTreeClassifier():

  def __init__(self):
    self._algo    = None # algorithm simple tree, ramdom forest, gradient boost, adpative boost.
    self._reader  = None # read data from csv or other format, and split subsamples for the training

  def set_reader(self, reader):
    self._reader = reader

  def set_algo(self, algorithm):
    self._algo = algorithm
    self._algo.set_reader(self._reader)

  def train(self,*args):
    self._algo.train(*args)

  def eval(self):
    self._algo.eval()




if __name__ == '__main__':
  import random
  random.seed(1)
  filename = 'data_banknote_authentication.csv'
  max_depth = 3
  min_size = 10
  nCut = 20
  r =DataReader()
  r.load_csv(filename)
  r.str_column_to_float()
  r.cross_validation_split()

  #run simple tree
  a = SimpleTree()
  dtc = DecisionTreeClassifier()
  dtc.set_reader(r)
  dtc.set_algo(a)
  dtc.train(max_depth, min_size, nCut)
  dtc.eval()


  #run random forest
  a = RandomForest()
  dtc = DecisionTreeClassifier()
  dtc.set_reader(r)
  dtc.set_algo(a)
  dtc.train(max_depth, min_size, nCut,20,0.5)
  dtc.eval()



  #run bdt
  a = BoostedDecisionTree()
  
  dtc = DecisionTreeClassifier()
  dtc.set_reader(r)
  dtc.set_algo(a)
  #dtc.train(max_depth, min_size, nCut,20,0.5)
  dtc.train(max_depth, min_size, nCut, 100, 0.5,'MSE')
  #dtc.train(max_depth, min_size, nCut, 100, 0.5,'MSE','adaptive')
  dtc.eval()



