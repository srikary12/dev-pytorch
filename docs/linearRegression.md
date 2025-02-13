# Linear Regression

## Purpose

- Helps find the relationship features and label
- Generally 2 dim, but can be scaled to any dimension
- Any statistical model which can fit a linear equation can be approximately classified by linear regression



## Implementation

```py
import kagglehub

# Download latest version from kaggle
path = kagglehub.dataset_download("hussainnasirkhan/multiple-linear-regression-dataset")
print("Path to dataset files:", path)

# Read csv
df = pd.read_csv(f"{path}" + "/multiple_linear_regression_dataset.csv", sep=",")

#EDA
import matplotlib.pyplot as plt
plt.plot(df["experience"], df["income"], 'ro')
plt.show()
```
![Plot showing data](img\linear reg\eda.png)
=== "Python"
    ```
    import numpy as np
    class linearRegressor:
        def __init__(self) -> None:
            self.w = 0.3 
            self.b = 0.4

        def forward(self, X):
            y_pred = [self.w*x + self.b for x in X]
            return y_pred

        def loss(self, y_pred, y_expected):
            y_pred = np.array(y_pred)
            y_expected = np.array(y_expected)
            return np.mean((y_expected-y_pred)**2)
        
        def backward(self, y, X, lr = 0.01):
            f = y - (self.w*X + self.b)
            N = len(X)
            self.w -= lr * (-2 * X.dot(f).sum() / N)
            self.b -= lr * (-2 * f.sum() / N)
            return self.w, self.b
    
    lr = linearRegressor()
    losses = []
    y_train  = df["income"]
    X_train  = np.array(df["experience"])
    y_pred = lr.forward(df["experience"])
    lr.loss(y_pred, df["income"])
    w,b = lr.backward(df["income"], df["experience"])
    for i in range(200):
        y_pred = lr.forward(X_train)
        loss = lr.loss(y_pred, y_train)
        lr.backward(y_train, X_train, lr=0.01)
        losses.append(loss)
    # Plot the learning process
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()
    ```
    ![Plot showing data](img\linear reg\loss.png)

    ```
    plt.plot(df["experience"], df["income"], 'ro')
    plt.plot(X_train, y_pred, color = "g")
    plt.show()
    ```
    ![Plot showing data](img\linear reg\pred.png)

=== "sklearn"
    

=== "Pytorch"