from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from regularization import *


# Regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = get_regression_data()

models_reg = [
    linear_regression(X_train_reg, y_train_reg),
    ridge_regression(X_train_reg, y_train_reg),
    lasso_regression(X_train_reg, y_train_reg),
]

for model in models_reg:
    y_pred = model.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, y_pred)
    print(f"Model: {type(model).__name__}, MSE: {mse:.2f}")

# Classification
X_train_clf, X_test_clf, y_train_clf, y_test_clf = get_classification_data()

models_clf = [
    logistic_regression(X_train_clf, y_train_clf),
    logistic_l2_regression(X_train_clf, y_train_clf),
    logistic_l1_regression(X_train_clf, y_train_clf),
]

for model in models_clf:
    y_pred = model.predict(X_test_clf)
    accuracy = accuracy_score(y_test_clf, y_pred)
    print(f"Model: {type(model).__name__}, Accuracy: {accuracy:.2f}")
    # Evaluate models
    def evaluate_models(models, X_test, y_test, metric_fn, metric_name):
        for model in models:
            y_pred = model.predict(X_test)
            metric_value = metric_fn(y_test, y_pred)
            print(f"Model: {type(model).__name__}, {metric_name}: {metric_value:.2f}")

    # Regression evaluation
    evaluate_models(models_reg, X_test_reg, y_test_reg, mean_squared_error, "MSE")

    # Classification evaluation
    evaluate_models(models_clf, X_test_clf, y_test_clf, accuracy_score, "Accuracy")
