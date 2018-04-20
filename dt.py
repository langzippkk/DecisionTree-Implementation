from __future__ import division
import matplotlib as mpl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
mpl.rc('figure', figsize=[12,8])  #set the default figure size
import itertools, random, math
import time

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data"
df = pd.read_csv(url, header = None, na_values="?")

dsmall = df.iloc[0:10, list(range(3)) + [279]]

class Node(object):
    def __init__(self, name, node_type, data, label=None, split=None):
        self.name = name
        self.node_type = node_type
        self.label = label
        self.data = data
        self.split = split
        self.children = []
        
    def __repr__(self):
        data = self.data
        if self.node_type != 'leaf':
            s = (f"{self.name} Internal node with {data[data.columns[0]].count()} rows; split" 
                f" {self.split.split_column} at {self.split.point:.2f} for children with" 
                f" {[p[p.columns[0]].count() for p in self.split.partitions()]} rows"
                f" and infomation gain {self.split.info_gain:.5f}")
        else:
            s = (f"{self.name} Leaf with {data[data.columns[0]].count()} rows, and label"
                 f" {self.label}")
        return s
                                    
class Split(object):
    def __init__(self, data, class_column, split_column, point=None):
        self.data = data
        self.class_column = class_column
        self.split_column = split_column
        self.info_gain = None
        self.point = point
        self.partition_list = None # stores the data points on each side of the split
        self.find_split_point()
        self.partitions()
    
    
    def compute_entropy(self, data):
        data = data.astype(int)
        #unique, count = np.unique(data, return_counts=True)
        count = np.bincount(data)
        count = count[count != 0]
        p = count / np.sum(count)
        return -np.sum(p * np.log2(p))
    
    def compute_info_gain(self, neg, pos):
        
        data = self.data[self.class_column].values.astype(int)
        H0 = self.compute_entropy(data)
        p_neg = len(neg) / len(data)
        p_pos = len(pos) / len(data)
        H_n = p_neg * self.compute_entropy(neg)
        H_p = p_pos * self.compute_entropy(pos)
        Ha = H_p + H_n
        return H0 - Ha
    
    def find_split_point(self):
        data = self.data[[self.split_column, self.class_column]].values
        attr_value = data[data[:,0].argsort()][:,0]
        idx = data[data[:,0].argsort()][:,-1]
        max_IG = -np.inf
        for i in range(len(attr_value) - 1):
            if attr_value[i] != attr_value[i + 1] and idx[i] != idx[i + 1]:
                
                split_point = (attr_value[i] + attr_value[i + 1]) / 2
                neg = idx[:i+1]
                pos = idx[i+1:]
                
                if self.compute_info_gain(neg, pos) > max_IG:
                    max_IG = self.compute_info_gain(neg, pos)
                    self.point = split_point
        self.info_gain = max_IG
    
    def partitions(self):
        '''Get the two partitions (child nodes) for this split.'''
        if self.partition_list:
            # This check ensures that the list is computed at most once.  Once computed
            # it is stored
            return self.partition_list
        data = self.data
        split_column = self.split_column
        partition_list = []
        partition_list.append(data[data[split_column] <= self.point])
        partition_list.append(data[data[split_column] > self.point])
        self.partition_list = partition_list


class DecisionTree(object):

    def __init__(self, max_depth=None):
        if (max_depth is not None and (max_depth != int(max_depth) or max_depth < 0)):
            raise Exception("Invalid max depth value.")
        self.max_depth = max_depth
        

    def fit(self, data, class_column):
        '''Fit a tree on data, in which class_column is the target.'''
        if (not isinstance(data, pd.DataFrame) or class_column not in data.columns):
            raise Exception("Invalid input")
            
        self.data = data
        self.class_column = class_column
        self.non_class_columns = [c for c in data.columns if c != class_column]
        self.root = self.recursive_build_tree(data, data, depth=0, attributes=self.non_class_columns, name='0')
  
    # Node __init__(self, name, node_type, data, label=None, split=None)
    def recursive_build_tree(self, data, parent_data, depth, attributes, name):
        
         if len(data) == 0: # data set is empty
            return Node(name=name, node_type='leaf', label=self.plurality_value(parent_data), 
                        data=data)
        
         elif depth == self.max_depth : # reach the max depth of the tree, can not split
            return Node(name=name, node_type='leaf', label=self.plurality_value(data), 
                        data=data)
         elif np.all(data[self.class_column].values == data[self.class_column].values[0]):
            return Node(name=name, node_type='leaf', label= list(set(data[self.class_column].values))[0], data=data)
        
         elif len(attributes) == 0: # only has the class column, no attribute
            return Node(name=name, node_type='leaf', label=self.plurality_value(data), 
                        data=data)
        #data[attributes].drop_duplicates()
         elif len(data[attributes].drop_duplicates()) == 1: # noise data
            return Node(name=name, node_type='leaf', label=self.plurality_value(data), 
                        data=data)
        
         else:
            split = None          
            for attribute in attributes: 
                temp_split = Split(data, self.class_column, attribute)     
                # set the split with higher info gain as the true split
                if not split or temp_split.info_gain > split.info_gain:
                    split = temp_split          
            root = Node(name=name, node_type='interval', data=data, split=split)
            non_class_columns = attributes
            
            if len(set(data[self.class_column].values)) == 2: # the attribute is discrete
                attributes = [c for c in attributes if c != root.split.split_column]
            
            # recursive_build_tree(self, data, parent_data, depth, attributes, name)
            root.children.append(self.recursive_build_tree(root.split.partition_list[0][attributes +[self.class_column]], data, depth + 1, attributes, name + '.0')) 
            root.children.append(self.recursive_build_tree(root.split.partition_list[1][attributes +[self.class_column]], data, depth + 1, attributes, name + '.1'))
            
            return root
    
    def predict(self, test):

        # WRITE YOUR CODE HERE
        res = []
        test = test.values
        for i in range(len(test)):
            node = self.root
            while node.node_type != 'leaf':
                split = node.split
                #if test[split.split_column][i] <= split.point:
                if test[i, split.split_column] <= split.point:
                    node = node.children[0]
                else:
                    node = node.children[1]
            res.append(node.label)
        return res
    
    def plurality_value(self, data):
        #return data[self.class_column].value_counts().idxmax()
        return np.argmax(np.bincount(data[self.class_column].astype(int).values))
                
    def print(self):
        self.recursive_print(self.root)
    
    def recursive_print(self, node):
        print(node)
        for u in node.children:
            self.recursive_print(u)  


tree = tree = DecisionTree(3)
tree.fit(dsmall, 279)
tree.print()

def validation_curve():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data"
    df = pd.read_csv(url, header = None, na_values="?")
    MAX_DEPTH = 20
    NUMBER_OF_COLUMNS = 278
    NUMBER_OF_ROWS = len(df) 
    
    CLASS_COLUMN = 279
    
    # fill the empty value
    for i in range(280):
        if df[i].isnull().sum() > 0:
            df.iloc[:,i].fillna(df[i].mode()[0], inplace=True)
    
    df = df.iloc[:NUMBER_OF_ROWS,list(range(NUMBER_OF_COLUMNS)) + [CLASS_COLUMN]]
    # shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    # split the data into 3 parts
    datasets = np.array_split(df, 3)
    
    # initilize the correct ratio of training data and test data
    training_error_ratio = []
    test_error_ratio = []
    for depth in range(MAX_DEPTH + 1)[2::2]:
        # initialize the tree
        dt = DecisionTree(depth)
        
        training_error_ratio_sum = 0
        test_error_ratio_sum = 0
        
        for sets in [[0,1,2],[1,2,0],[0,2,1]]:
            # get the training data and test data
            training_data = pd.concat([datasets[sets[0]],datasets[sets[1]]])
            test_data = datasets[sets[2]]
            # train the model
            dt.fit(training_data, CLASS_COLUMN)
            # get the prediction result of the training data and test data
            training_result_list = dt.predict(training_data)
            test_result_list = dt.predict(test_data)
            
            training_error_ratio_sum += 1 - np.sum(training_result_list == 
                                                 training_data[CLASS_COLUMN]) / len(training_data)
            test_error_ratio_sum += 1 - np.sum(test_result_list == test_data[CLASS_COLUMN]) / len(test_data)
        training_error_ratio.append(training_error_ratio_sum / 3)
        test_error_ratio.append(test_error_ratio_sum / 3)
        
        print('layers ' + str(depth))
        
    x = range(MAX_DEPTH + 1)[2::2]
    plt.ylabel("$error \ ratio$")
    plt.xlabel("$depth$")
    plt.plot(x, training_error_ratio, label='training')
    plt.plot(x, test_error_ratio, label='test')
    plt.legend()
start = time.time()
validation_curve()
end = time.time()
print(end - start)
