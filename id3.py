'''
Implementation of ID3 algorithm
Author - Shreyas Kolpe
Date - 2/7/2018
'''
import pandas as pd 
import math
import copy
import operator

# Function to calculate entropy given a dictionary of labels and counts
def entropy(classification):
    values = list(classification.values())
    norm = sum(values)
    entr = 0.0
    for val in values:
            if val == 0:
                    continue
            entr = entr - (val/norm)*math.log(val/norm,2)
    return (entr,norm)

# Function to calculate entropy as weighted sum of entropies of the divisions formed by splitting on an attribute
def attr_entropy(count_dict, attribute_dict, attr):
    count_attr = count_dict[attr]
    norm = 0
    entr = 0
    for val in attribute_dict[attr]:
        (e,n) = entropy(count_attr[val])
        entr = entr + n*e
        norm = norm + n
    entr = entr/norm
    return entr

# Class that defines a node in the decision tree
class TreeNode:
    'Defines one node in the decision tree'

    def __init__(self, df):
        self.df = df
        self.attribute_dict = {}
        self.split_attr, self.leaf = self.split()
        self.children = {}
        # If not a leaf, create children
        if(self.leaf == False):
            for val in self.attribute_dict[self.split_attr]:
                # Filter out parent's dataframe before sending to child
                query = "{} == '{}'".format(self.split_attr, val)
                dataframe = df.query(query)
                del dataframe[self.split_attr]
                self.children[val] = TreeNode(dataframe)

    def split(self):
        # Frist find out what values are taken by the attributes
        self.attributes = list(self.df.columns.values)
        for attr in self.attributes:
            self.attribute_dict[attr] = list(self.df[attr].unique())
        label = self.attributes.pop()

        # Then count how many rows correspond to each value of the label/target variable
        label_count = {}
        for labelval in self.attribute_dict[label]:
            label_count[labelval] = len(self.df.query("{} == '{}'".format(label, labelval)))

        # If there are no attributes left, return majority class
        if(len(self.attributes) == 0):
            return max(label_count.items(), key=operator.itemgetter(1))[0] , True

        # Calculate entropy before split    
        current_entropy = entropy(label_count)

        # If entropy is 0, all data points have same class label and return that label
        if(current_entropy[0] == 0.0):
            return max(label_count.items(), key=operator.itemgetter(1))[0] , True

        # Otherwise, create a count for all combinations of attribute values and target variable values    
        count_dict = {}
        for attr in self.attributes:
            count_dict[attr] = {}
            for val in self.attribute_dict[attr]:
                count_dict[attr][val] = {}
                for labelval in self.attribute_dict[label]:
                    # Create a filter to apply as a query
                    filter_dict = {attr: val, label: labelval}
                    query = ' & '.join(["{} == '{}'".format(k,v) for k,v in filter_dict.items()])
                    count_dict[attr][val][labelval] = len(self.df.query(query))

        # Calculate IG for each attribute and find the one with maximum
        split_entropy = {attr: (current_entropy[0] - attr_entropy(count_dict, self.attribute_dict, attr)) for attr in self.attributes}
        max_attr = self.attributes[0]
        for attr in self.attributes[1:]:
            if split_entropy[attr] > split_entropy[max_attr]:
                max_attr = attr

        # If all IG are uniformly 0, return majority class
        if split_entropy[max_attr] == 0.0:
            return max(label_count.items(), key=operator.itemgetter(1))[0] , True

        # Otherwise, return max
        return max_attr, False

    def traverse(root, spaces):

        # Print and traverse in DFS
        print(root.split_attr)
        if(root.leaf == True):
            return
        else:
            for key in root.children.keys():
                for i in range(2*spaces):
                    print(' ',end='')
                print(key,end=' : ')
                TreeNode.traverse(root.children[key], spaces+1)


    def predict(root, data_dict):

        # Follow the tree to predict
        if(root.leaf == True):
            return root.split_attr
        else:
            return TreeNode.predict(root.children[data_dict[root.split_attr]], data_dict)


# Reading the data
df = pd.read_csv('dt-data.txt', sep=', ', engine='python')

# Removing ( and ) characters from first and last attribute names
attributes = list(df.columns.values)
attributes[0] = attributes[0][1:]
attributes[-1] = attributes[-1][:-1]

df.columns = attributes

# Removing : and ; characters from first and last columns
for i in range(len(df.index)):
    df.iloc[i, df.columns.get_loc(attributes[0])] = df.iloc[i, df.columns.get_loc(attributes[0])].split(": ")[1]
    df.iloc[i, df.columns.get_loc(attributes[-1])] = df.iloc[i, df.columns.get_loc(attributes[-1])][:-1]

#To remove space in column names
cols = df.columns
cols = cols.map(lambda x: x.replace(' ','_'))
df.columns = cols

# create decision tree, done recursively within
root = TreeNode(df)

print("\n The decision tree generated by ID3 :\n")
TreeNode.traverse(root, 1)

data_dict = {'Occupied':'Moderate', 'Price':'Cheap', 'Music':'Loud', 'Location':'City-Center', 'VIP':'No', 'Favorite_Beer':'No'}

print("\n Given the data ")
print(data_dict)
print("the prediction for 'Enjoy' : "+TreeNode.predict(root,data_dict))