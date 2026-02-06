# California Housing Price Prediction
Quick access:
- [Abstract](#abstract)
- [How to run](#abstract)
- [Quick Look on The Dataset](#quick-look-on-the-dataset)
- [Comparing Models Performance](#model-performance-comparison)
  - [Models Performance Table](#models-performance-table)

## Abstract
This project implements linear and polynomial regression models to predict California housing prices. By comparing performance metrics across multiple degrees of complexity, I identify the optimal model for generalization.

- Dataset: sklearn's california_housing, [More info](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- Programming Language: Python
- libraries & dependencies: scikit-learn, numpy

## How to Run
`LinearRegression.ipynb` contains the linear model, and `PolynomialRegression.ipynb` contains the polynomial model.

Make sure to run all previous cells in the notebook before using the prediction snippet to ensure the model and custom functions are loaded into memory.

You can change the degree of the polynomial model by changing the `DEGREE` variable in `Global Variables` Section in the notebook.

There is a custom `predict` function in each notebook. To see model's predicted value for one specific sample, just add the code below at the end of the notebook (Replace the words with numbers according to your sample) :
```
x = np.array([MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude])
print(predict(x))
```
The `predict` function handles scaling and reshaping itself, just give it the raw data and you will be fine.

To obtain model's predicted values for a whole dataset, save your dataset into a variable(lets say `Data`) as a numpy matrix, and then insert the code below at the end of the notebook:

For polynomial regression:
```
Data = poly.transform(Data)
Data = scaler.transform(Data)
print(model.predict(Data))
```

For linear regression:
```
Data = scaler.transform(Data)
print(model.predict(Data))
```

The result will be a numpy array.

## Quick Look on The Dataset
Dataset has eight columns as independent variables and one dependent variable(price), which we are asked to predict. all values are continuous, Hence, using a regression model looks as the best idea.

| Metric | Value |
| :--- | :--- |
| **Samples Total** | 20,640 |
| **Dimensionality** | 8 |
| **Features** | Real |
| **Target Range** | Real (0.15 – 5.0) |

## Comparing Models Performance
For this data set, a total of four regression models were tested; A linear regression model and three Polynomial regression models, with degrees two, three and four.

All models have been trained on the same training test(subset of the whole dataset with random_state=24), and the same test set.

### Models Performance Table

| Model | Linear | poly. Deg=2 | poly. Deg=3 | poly. Deg=4 |
| :--- | :--- | :--- | :--- | :--- |
| **RMSE \| Training** | 0.7196 | 0.6486 | 0.5848 | 0.5306 |
| **RMSE \| Test** | 0.7455 | 0.6813 | 5.0405 | 122.6364 |
| **Test / Training** | 1.0359 | 1.0505 | 8.6191 | 231.09 |

As obvious, for polynomial regressions with degree above two, error on training test decreases, but error on the test set increases much more, leaving us with a very high proportion of test error over training error. This means the the models work very well on the data they've been trained, but very bad on a data they have never seen before. This is an obvious sign of Overfitting, for degrees exceeding two.

This leaves us with two models, the linear model and the degree two polynomial model. The polynomial model has a Test/Training of 1.05, which means this model generalizes very well and can maintain its accuracy over unseen data(No overfitting). Also, it has less RMSE than the linear model, it's more accurate on both training and test data. Hence, polynomial model seems to be the best.