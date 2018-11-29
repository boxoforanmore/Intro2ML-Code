import pandas as pd

#web_site = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
#print(web_site)

df = pd.read_csv('./housing.data', sep='\s+')

# Feature columns in above Housing data set with descriptions
col_2_desc = {'CRIM'    : 'Per capita crime rate per town',
              'ZN'      : 'Proportion of residential land zoned for lots over 25,000 sq. ft.',
              'INDUS'   : 'Proportion of non-retail business acres per town',
              'CHAS'    : 'Charles River dummy variable (=1 if tract bounds river; 0 otherwise)',
              'NOX'     : 'Nitric oxide concentration (parts per 10 million)',
              'RM'      : 'Average number of rooms per dwelling',
              'AGE'     : 'Proportion of owner-occupied units built prior to 1940',
              'DIS'     : 'Weighted distances to five Boston employment centers',
              'RAD'     : 'Index of accessibility to radial highways',
              'TAX'     : 'Full-value property tax rate per $10,000',
              'PTRATIO' : 'Pupil-teacher ratio by town',
              'B'       : '1000(Bk-0.63)^2, where Bk is the proportion of [people of African American Descent] by town',
              'LSTAT'   : 'Percentage of lower status of the population',
              'MEDV'    : 'Median value of owner-occupied homes in $1000s'}

# Load up columns
columns = []
for col in col_2_desc:
    columns.append(col)

df.columns = columns

# Check the first 5 lines of the dataset
print(df.head())
print()
print()


# Create and visualize scatterplot matrix
import matplotlib.pyplot as plt
import seaborn as sns
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']

# Can show the full scatterplot matrix; the chosen is used for space constraints
# size is deprecated; use height instead for graph size
sns.pairplot(df[cols], height=2)
plt.tight_layout()
plt.show()



# Plot correlation matrix as heatmap of Pearson's r
import numpy as np

cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True,
                 fmt='.2f', annot_kws={'size':15},
                 yticklabels=cols,
                 xticklabels=cols)

plt.show()



# Using the LinearRegressionGD regressor file
from LinRegGD import LinearRegressionGD

# RM has the highest correlation with MEDV
X = df[['RM']].values
y = df['MEDV'].values

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
sc_y = StandardScaler()

X_std = sc_x.fit_transform(X)

# Lets the transformer see and process the data as the expected 2-D array
# but flattens back to a 1-D array after the fit
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

lr = LinearRegressionGD()
lr.fit(X_std, y_std)

# Plot the cost as a function over the number of epochs 
# to make sure the algorithm is converging to a minimum
sns.reset_orig()
plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()



# Plot scatterplot of samples and the linear regression line
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
    return None


# Plots number of rooms agains housing prices
lin_regplot(X_std, y_std, lr)
plt.xlabel(str(col_2_desc['RM'] + ' [RM] (standardized)'))
plt.ylabel(str(col_2_desc['MEDV'] + ' [MEDV] (standardized)'))
plt.show()
