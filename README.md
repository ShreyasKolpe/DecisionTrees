# DecisionTrees

The data consists of the recordings of whether a person enjoyed themselves on an evening out in Jerusalem.

There are several features, and one target feature - Enjoy

For example,
Occupied: High, Price: Expensive, Music: Loud, Location: Talpiot, VIP: No, Favorite Beer: No, Enjoy: No

The task is to build an ID3 decision tree to predict if the person would enjoy given the following data

Occupied: Moderate, Price: Cheap, Music: Loud, Location: City-Center, VIP: No, Favorite Beer: No


## Implementation

The program id3.py is structured as follows –
** Methods to calculate entropy –
entropy(classification) – calculates entropy for any split of the dataset based on the counts of the values taken by the target variable. It returns a tuple - entropy and the total count (to be used as a weighting factor). The argument classification is a dictionary that keeps this count for the variable ‘Enjoy’. For instance,
                         {‘Yes’: 3 , ‘No’: 2}
attr_entropy(count_dict, attribute_dict, attr) – calculates and returns the weighted sum of entropy values for the dataset that would result if we split on the attribute attr.
  
Of the other two arguments –attribute_dict is a dictionary that stores all the
values taken by all the attributes in the dataset.
For instance,
{ ‘Occupied’: [‘High’, ‘Moderate’, ‘Low’], ‘Price’: [‘Expensive’, ‘Normal’, ‘Cheap’] ... }
count_dict is also a dictionary that keep the count values of the target variable for each of the values taken by all the attributes.
For instance,
{‘Occupied’: {‘High’: {‘Yes’: 2, ‘No’: 5},
                             ‘Low’: {‘Yes’: 1,
                                     ‘No’: 2},
...},
...}
** The class TreeNode that represents a single node in the decision tree
** The code that loads the data initially and calls the method to create a decision tree.
Here, the data from dt-data.txt is loaded into a pandas DataFrame, which is the data structure that will be used throughout the program to represent the data.
  
Initially, some preprocessing is done to make the data more amenable for analysis – removing parentheses from attribute names (headers), removing white spaces and other characters from the attribute names and the data values.
After the processing, the root of the tree is instantiated by calling the TreeNode() constructor and passing the DataFrame object to it.
This constructor finds the attribute to split on at the root level, and then recursively creates its children. This process continues until all the data points are classified homogeneously or there are no attributes left to split on.
The constructor calls split() which returns a tuple – a bool value that signifies if the node is a leaf node, and a string which is the classification if it is a leaf node and the attribute to split on if it is not a leaf node.
The split() method is where the meat of the matter lies.
It first creates and populates attribute_dict – which is a dictionary of all the values taken by all the attributes. It also creates the dictionary label_count – which counts the number of occurrences for each of the values of the target variable in the DataFrame. The entropy of the data as given is calculated using label_count as current_entropy.
If there are no attributes left to split on, or if the current_entropy is 0, the method returns the value of the target variable that occurs most frequently as the classification.
Otherwise, count_dict is created with all the combinations of values taken by all the attributes in the DataFrame and the target variable as the keys. The values are populated by querying the DataFrame for the particular attribute value and target variable value.
Then, the information gain is calculated for each attribute as the difference of the current_entopy[0] and attr_entropy() called with each attribute. The attribute that gives the highest information gain is returned.
Optimization – In the scenario where all the attributes have 0 information gain on split, it is evident that there is no use in creating further split. The program

handles this situation by returning the value of the target variable that occurs most frequently as the classification and terminating this branch of the tree. This occurs in one case in the program run on the given dataset.
Once split() returns the attribute to split on if it is not a leaf, the constructor creates children as TreeNode objects for each value of that attribute in the DataFrame. This is done by passing a DataFrame object from which this attribute column is removed and the rows have been filtered by applying the particular value of the attribute. Thus, the recursion continues.
The traverse() method then recursively travels down the tree from the root in a depth-first manner and prints out the tree.
The predict() method traverses the same tree and prints the prediction for the data point given passed to it as a dictionary of attribute and value pairs.