# DataCleansing
Demonstrate data formatting/cleaning for exploration by data analysts for model building.
The following file should be downloaded into the project directory and unzipped before executing code
https://www.kaggle.com/tmdb/tmdb-movie-metadata/downloads/tmdb-5000-movie-dataset.zip


Objectives:
1. Format data to the required input format of the machine learning algorithm used
2. Ensure data quality

Data preprocessing includes data cleaning, normalization, transformation, feature extraction and selection.
The output of data preprocessing is the final input set for the machine learning algorithm.

# Instance selection/outlier detection:
Records with too many nulls for feature values should be removed from the training set as they represent outliers
There are two main methods used for instance selection/outlier detection:
Filter method : filter out outliers
Wrapper method : uses an ML algorithm to trigger instance selection

Variable by variable data cleaning is a straightforward filter approach:

eg: Illegal values in the data can be detected by checking for data with

	1. cardinality issues eg(count of distinct genders >2)
	2. not within established min,max values
	3. variance/deviation issues : variance and deviation of statistical values cannot be greater than a threshold

The filter based approach does not detect inliers (an erroneous data value that lies inside a statistical distribution)
In that case a more involved approach (wrapper method) would be necessary
eg:

	1. distance based outlier detection algorithm 
	2. density based outliers

# Fixing missing feature values:
Missing feature values:
Occurs due to:

	1. data loss
	2. feature does not exist for certain instance types
	3. for a given result, the feature is irrelevant
  
Solutions:

	1. ignore instances with unknown feature values
	2. Most common feature value : (default unknown to most common feature)
	3. Concept most common feature value : (more intelligently default unknowns to most common feature value for the particular class of instances)
	4. Mean substitution: substitute either general mean or more intelligently mean of the particular class of which instance is a member
	5. Develop a classification model with the unknown feature as target and all other features as predictors
	6. Hot deck impudation : identify an instance most similar to the current instance and substitute missing feature value from that instance
	7. Method of treating unknown itself as a special value in the model
  

# Discretization:

Individual numerical values that are continuous should be grouped into classes. This is to reduce the number of features to improve perfomance of the model.

The simplest discretization method is an unsupervised direct method named equal size discretization. It calculates the maximum and the minimum for the feature that is being discretized and partitions the range observed into k equal sized intervals. 

Equal frequency is another unsupervised method. It counts the number of values we have from the feature that we are trying to discretize and partitions it into intervals containing the same number of instances.

Discretization methods can be performed in either a top down or a bottom up fashion

# Data normalization:
if there is a huge difference between minimum and maximum values in a feature, it should be normalized and the values scaled to appreciably low values.

The common methods for these are:

  min-max normalization:
  v' = (v-minA)/(maxA-minA) * (new_maxA - new_minA) + new_minA

  z-score normalization: 
  v' = v-meanA/std_devA

where v is the old feature value and v' the new one.

# Feature subset reduction:

Feature subset selection is the process of identifying and removing as much irrelevant and redundant features as possible
Reduces dimensionality of the data and allows the ML algorithm to operate faster and more effectively

Features are classified as:

	1. Relevant: have an influence on the output and role cannot be assumed by other features
	2. Irrelevant : not having an influence on the output and whose values are generated at random
	3. Redundant: a relevant or irrelevant feature whose role can be taken over by another feature

# Feature construction:

The problem of feature interaction can be also addressed by constructing new features from the basic feature set. This technique is called feature construction/transformation.


Reference : Data Preprocessing for Supervised Learning Whitepaper [S. B. Kotsiantis, D. Kanellopoulos and P. E. Pintelas]
