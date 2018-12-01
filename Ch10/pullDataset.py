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



# Scaling the predicted price outcome back to the original scale
# with the inverse transform method of StandardScaler
## This checks the price of a house with 5 rooms


### The book has errata here
num_rooms_std = sc_x.transform(np.array(5.0).reshape(1,-1))
price_std = lr.predict(num_rooms_std)
print()
print("Price in $1000s for 5 room house: %.3f" % sc_y.inverse_transform(price_std))
print()

num_rooms_std = sc_x.transform(np.array(100.0).reshape(1, -1))
price_std = lr.predict(num_rooms_std)
print()
print("Price in $1000s for 100 room house: %.3f" % sc_y.inverse_transform(price_std))
print()


## The weights of the intercept do not need to be updated with standrdized variables
print()
print('Slope: %.3f' % lr.w_[1])
print()
print('Intercept: %.3f' % lr.w_[0])


# Estimating coefficient of a regression model via sklearn
# Sklearn tends to use LIBLINEAR libraries for vectorized optimation algorithms
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X, y)
print('Slope: %.3f' % slr.coef_[0])
print()
print('Intercept: %.3f' % slr.intercept_)
print()
print()


lin_regplot(X, y, slr)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.show()


# Closed form solution without ML library
# adding a column vector of 'ones'
Xb = np.hstack((np.ones((X.shape[0], 1)), X))
w = np.zeros(X.shape[1])
z = np.linalg.inv(np.dot(Xb.T, Xb))
w = np.dot(z, np.dot(Xb.T, y))

print()
print('Slope: %.3f' % w[1])
print()
print('Intercept: %.3f' % w[0])
print()


# Using RANSAC for a more robust algorithm
# Looks at inliers instead of throwing out outliers
from sklearn.linear_model import RANSACRegressor

# Set max iterations to 100, minimum number of random samples to 50
# Residual threshold--only allows samples to be included in inliers
# if their veritical distance is less than or equal to 5 distance units
ransac = RANSACRegressor(LinearRegression(), max_trials=100,
                         min_samples=50, loss='absolute_loss',
                         residual_threshold=5.0,
                         random_state=0)

ransac.fit(X, y)


# Plot the linear fit of ransac
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])

plt.scatter(X[inlier_mask], y[inlier_mask], c='steelblue', edgecolor='white',
            marker='o', label='Inliers')

plt.scatter(X[outlier_mask], y[outlier_mask], c='limegreen', edgecolor='white',
            marker='s', label='Outliers')

plt.plot(line_X, line_y_ransac, color='black', lw=2)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper left')
plt.show()


print()
print('Ransac Slope: %.3f' % ransac.estimator_.coef_[0])
print()
print('Ransac Intercept %.3f' % ransac.estimator_.intercept_)
print()


# Evaluating the performance of a linear regression model
from sklearn.model_selection import train_test_split

X = df.iloc[:, :-1].values
y = df['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white',label='Training Data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test Data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')

plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
plt.xlim([-10, 50])
plt.show()



# Computing the MSE (mean standard error)
from sklearn.metrics import mean_squared_error

print()
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred),
                                       mean_squared_error(y_test, y_test_pred)))
print()


# Coefficient of determination (Standardized MSE)
from sklearn.metrics import r2_score

print()
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred),
                                       r2_score(y_test, y_test_pred)))
print()


# Ridge regularization model initialization (reg strength must be initialized)
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)


# LASSO initialization (reg strength must be initialized)
from  sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0)


# ElasticNet initialization (allows playing with L1/L2 ratio)
from sklearn.linear_model import ElasticNet

# l1_ratio of 1 will make the regressor equal to LASSO
elanet = ElasticNet(alpha=1.0, l1_ratio=0.5)





# Modeling nonlinear relationships in the Housing dataset
# between LSTAT and housing prices
from sklearn.preprocessing import PolynomialFeatures

X = df[['LSTAT']].values
y = df['MEDV'].values

regr = LinearRegression()

# create quadratic features 
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

# fit features
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))

plt.scatter(X, y, label='training points', color='lightgray')

plt.plot(X_fit, y_lin_fit, label='linear (d=1), $R^2=%.2f$' % linear_r2,
         color='blue', lw=2, linestyle=':')

plt.plot(X_fit, y_quad_fit, label='quadratic (d=2), $R^2=%.2f$' % quadratic_r2,
         color='red', lw=2, linestyle='-')

plt.plot(X_fit, y_cubic_fit, label='cubic (d=3), $R^2=%.2f$' % linear_r2,
         color='green', lw=2, linestyle='--')


plt.xlabel('LSTAT : ' + col_2_desc['LSTAT'])
plt.ylabel('MEDV : ' + col_2_desc['MEDV'])
plt.legend(loc='upper right')
plt.show()



# Testing to see if a logistic expression of a function with a negative right side may work better 
# (ie a decreasing exponential function)
X_log = np.log(X)
y_sqrt = np.sqrt(y)


# fit the features
X_fit = np.arange(X_log.min()-1, X_log.max()+1, 1)[:, np.newaxis]
regr = regr.fit(X_log, y_sqrt)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y_sqrt, regr.predict(X_log))


plt.scatter(X_log, y_sqrt, label='training points', color='lightgray')
plt.plot(X_fit, y_lin_fit, label='linear (d=1), $R^2=%.2f$' % linear_r2,
         color='blue', lw=2, linestyle='--')

plt.xlabel('log(% lower status of the population [LSTAT])')
plt.ylabel('$\sqrt{Price \; in \; \$1000s \; [MEDV]}$')
plt.legend(loc='lower left')
plt.show()


# Print r2 logistic transformation:
print()
print('Logistic Transformation R^2=%.2f' % linear_r2)
print()



# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor

X = df[['LSTAT']].values
y = df['MEDV'].values
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)

sort_idx = X.flatten().argsort()
lin_regplot(X[sort_idx], y[sort_idx], tree)

plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')
plt.show()



# Random forest regression
# 60% training set / 40% test set
from sklearn.ensemble import RandomForestRegressor

X = df.iloc[:, :-1].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

forest = RandomForestRegressor(n_estimators=1000, criterion='mse', random_state=1, n_jobs=-1)
forest.fit(X_train, y_train)

y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

print()
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), \
                                       mean_squared_error(y_test, y_test_pred)))
print()
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), \
                                       r2_score(y_test, y_test_pred)))
print()

plt.scatter(y_train_pred, y_train_pred - y_train,
            c='steelblue', edgecolor='white',
            marker='o', 
            s=35,
            alpha=0.9,
            label='Training data')

plt.scatter(y_test_pred, y_test_pred - y_test,
            c='limegreen', edgecolor='white',
            marker = 's',
            s=35,
            alpha=0.9,
            label='Test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='black')
plt.lim([-10, 50])
plt.show()
