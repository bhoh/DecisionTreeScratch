from random import seed
from random import randrange
from csv import reader

class DataReader():

  def __init__(self):
    pass

  # Load a CSV file
  def load_csv(self, filename):
    file = open(filename, "rt")
    lines = reader(file)
    self.dataset = list(lines)

  # Convert string column to float
  def str_column_to_float(self):
    for column in range(len(self.dataset[0])):
      for row in self.dataset:
        try:
          row[column] = float(row[column].strip())
        except:
          row[column] = -9999999

  def remove_column_row(self,column_idx,row_idx):
    self.dataset.pop(row_idx)
    for row in self.dataset:
      row.pop(column_idx)

  # Split a dataset into k folds
  def random_split(self, dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
      fold = list()
      while len(fold) < fold_size:
        index = randrange(len(dataset_copy))
        fold.append(dataset_copy.pop(index))
      dataset_split.append(fold)

    return dataset_split

  # Split a dataset into train and test dataset
  def cross_validation_split(self, ratio_train=0.7):
    len_dataset = len(self.dataset)
    dataset_split = self.random_split(self.dataset, int(len_dataset/(ratio_train*len_dataset)*100))
    self.dataset_train = [ data for fold in dataset_split[:100] for data in fold ]
    self.dataset_test  = [ data for fold in dataset_split[100:] for data in fold]
    #print(self.dataset_train)

  # Split a train dataset into k folds
  def split_train_samples(self, n_folds):
    self.dataset_train_split = self.random_split(self.dataset_train, n_folds)
  
if __name__ == '__main__':
  
  filename = 'data_banknote_authentication.csv'
  n_folds = 5
  r =DataReader()
  r.load_csv(filename)
  r.str_column_to_float()
  r.cross_validation_split()
  r.split_train_samples(5)
