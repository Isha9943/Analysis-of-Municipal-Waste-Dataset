# importing libraries and packages
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error,r2_score

import warnings
warnings.filterwarnings("ignore")


# Reading X from file
data = pd.read_csv('public_data_waste_fee.csv')
print(data.info(),'\n\n')
# Replacing data in numeric columns with median
data = data.fillna(data.median(numeric_only=True))
# As the data column 'name' has large number of unique categorical values, therefore, dropping 'name' column as
data.drop('name',axis=1,inplace=True)
# Dropping the tuples with missing values
data.dropna(inplace=True)
print(data.info(),'\n\n')

# Separating dependent variables from target variables
target = 'tc'
X = data.drop(target,axis=1)
Y = data[target]

# Separating type of attributes in X (Numerical (Continuous and Discrete) & Categorical)
num_cols = X.select_dtypes(exclude='object').columns
cat_cols = X.select_dtypes(include='object').columns
dis_cols = [col for col in num_cols if X[col].unique().sum()<10]
con_cols = [col for col in num_cols if col not in dis_cols]

print("Numerical Attributes: ", list(num_cols))
print("Categorical Attributes: ", list(cat_cols))
print("Numerical Discrete Attributes: ", list(dis_cols))
print("Numerical Continuous Attributes: ", list(con_cols))

# Checking the distribution of the continuous attributes
for col in con_cols:
    plt.hist(X[col],bins=20)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.title(f"{col} distribution plot")
    plt.show()

# Details of continuous attributes:
for col in con_cols:
    print(f'\n\n----{col}----')
    print("Mean:", X[col].mean())
    print("Median:", X[col].median())
    print("Mode:", X[col].mode())
    print("Minimum:", X[col].min())
    print("Maximum:", X[col].max())
    print("Standard Deviation:", X[col].std())

# # As the data is highly skewed, using logarithmic transformation to remove skewness
for col in num_cols:
    if 0 in X[col].unique():
        X[col] = np.log1p(X[col])
    else:
        X[col] = np.log(abs(X[col]))

# Boxplots to check for outliers
for col in con_cols:
    plt.boxplot(X[col])
    plt.xlabel(col)
    plt.ylabel(col)
    plt.title(f"Boxplot for {col}")
    plt.show()


# Outlier Analysis: Dropping tuples with outliers (Outliers are the values beside 2.7 * std from mean)
index = list(X.index)
for col in con_cols:
    mean = X[col].mean()
    std = X[col].std()
    lb = mean-2.7*std
    ub = mean+2.7*std
    for j in index:
        if X[col][j]>ub or X[col][j]<lb:
            X[col][j] = np.nan
X.dropna(inplace=True)
Y = Y.filter(items=X.index)

# Finding correlation of num_cols with target column
for col in num_cols:
    print(f"Correlation of {col} with tc is: {np.round(X[col].corr(Y),4)}")


# As the data columns wden, pop, alt and plastic are almost uncorrelated with the target column 'tc' (<0.05)
# Therefore, dropping these X columns.
drop = ['alt','pden','wden','plastic']
X = X.drop(columns=drop,axis=1)
num_cols = X.select_dtypes(exclude='object').columns

# Standardizing the numerical data
scaler = StandardScaler()
scaler.fit_transform(X[num_cols])

# TargetEncoder for categorical data values
for col in cat_cols:
    labels_ordered=data.groupby([col])['tc'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    X[col]=X[col].map(labels_ordered)



# Dividing the data into test and train datasets with 20/80 split respectively
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.20,random_state=42,shuffle=True)
print()
print(f"X_train.shape: {X_train.shape}")
print(f"X_test.shape: {X_test.shape}")
print(f"Y_train.shape: {y_train.shape}")
print(f"y_test.shape: {y_test.shape}")
print()


# Linear Regression Model
lr = LinearRegression()
lr.fit(X_train,y_train)
pred1 = lr.predict(X_test)

# Regression Plot for Linear Regression Model
fig = sns.regplot(y_test,pred1)
fig.set(xlabel='True value',ylabel='Predicted Value',title='Linear Regression Model')
plt.show()

# Checking the prediction quality of the model
print(f"R2 score for Linear Regression Model: {r2_score(y_test,pred1)}")
print(f"Mean Squared Error of Linear Regression Model: {mean_squared_error(y_test,pred1)}")
print(f"Root Mean Squared Error of Linear Regression Model: {mean_squared_error(y_test,pred1)**0.5}")

# Residual Error Plot
plt.scatter(lr.predict(X_train),lr.predict(X_train)-y_train,color='blue',label='Train Data')
plt.scatter(lr.predict(X_test),lr.predict(X_test)-y_test,color='red',label='Test Data')
plt.legend(loc='upper right')
plt.hlines(y=0,xmin=0,xmax=600)
plt.xlabel("Predicted Value")
plt.ylabel('Error')
plt.title("Residual Error for Linear Regression Model")
plt.show()

# Random Forest Regressor Model
rf = RandomForestRegressor(n_estimators=1000,random_state=42)
model2 = rf.fit(X_train,y_train)
pred2 = rf.predict(X_test)


# Regression Plot for RandomForestRegressor Model
fig = sns.regplot(y_test,pred2)
fig.set(xlabel='True value',ylabel='Predicted Value',title='Random Forest Regression Model')
plt.show()

# Checking the prediction quality of the model
print(f"R2 score for Random Forest Regression Model: {r2_score(y_test,pred2)}")
print(f"Mean Squared Error of Random Forest Regression Model: {mean_squared_error(y_test,pred2)}")
print(f"Root Mean Squared Error of Random Forest Regression Model: {mean_squared_error(y_test,pred2)**0.5}")

# Residual Error Plot
plt.scatter(rf.predict(X_train),rf.predict(X_train)-y_train,color='blue',label='Train Data')
plt.scatter(rf.predict(X_test),rf.predict(X_test)-y_test,color='red',label='Test Data')
plt.legend(loc='upper right')
plt.hlines(y=0,xmin=0,xmax=600)
plt.xlabel("Predicted Value")
plt.ylabel('Error')
plt.title("Residual Error for Random Forest Regression Model")
plt.show()

# HistGradientBoostingRegression Model
reg = HistGradientBoostingRegressor(max_iter=500)
model3 = reg.fit(X_train,y_train)
pred3 = reg.predict(X_test)


# Regression Plot for HistGradientBoosting Regression Model
fig = sns.regplot(y_test,pred3)
fig.set(xlabel='True value',ylabel='Predicted Value',title='HistGradientBoosting Regression Model')
plt.show()

# Checking the prediction quality of the model
print(f"R2 score for HistGradientBoosting Regression Model: {r2_score(y_test,pred3)}")
print(f"Mean Squared Error of HistGradientBoosting Regression Model: {mean_squared_error(y_test,pred3)}")
print(f"Root Mean Squared Error of HistGradientBoosting Regression Model: {mean_squared_error(y_test,pred3)**0.5}")

# Residual Error Plot
plt.scatter(reg.predict(X_train),reg.predict(X_train)-y_train,color='blue',label='Train Data')
plt.scatter(reg.predict(X_test),reg.predict(X_test)-y_test,color='red',label='Test Data')
plt.legend(loc='upper right')
plt.hlines(y=0,xmin=0,xmax=800)
plt.xlabel("Predicted Value")
plt.ylabel('Error')
plt.title("Residual Error for HistGradientBoosting Regression Model")
plt.show()





