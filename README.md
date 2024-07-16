
# Different Types of Imputer

This repository contains a Jupyter notebook demonstrating different types of imputers for handling missing data. The notebook includes examples of using KNNImputer, IterativeImputer, and a deep learning-based imputer using TensorFlow/Keras.

## Requirements

To run the notebook, you need to have the following libraries installed:
- pandas
- numpy
- scikit-learn
- tensorflow

You can install the required libraries using the following command:
```sh
pip install pandas numpy scikit-learn tensorflow
```

## Usage

The provided notebook demonstrates how to handle missing values using various imputation methods. Below is a summary of the steps for each imputer used in the notebook.

### KNNImputer

1. **Import the necessary libraries:**
    ```python
    from sklearn.impute import KNNImputer
    import pandas as pd
    import numpy as np
    import random
    ```

2. **Create sample data with missing values:**
    ```python
    train = pd.DataFrame({
        'zero': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    column = train['zero'].copy()
    missing_pct = int(column.size * 0.4)
    i = [random.choice(range(column.shape[0])) for _ in range(missing_pct)]
    column.iloc[i] = np.NaN
    train['zero'] = column
    ```

3. **Apply KNNImputer:**
    ```python
    imputer = KNNImputer(n_neighbors=2, weights='uniform')
    train[['zero']] = imputer.fit_transform(train[['zero']])
    ```

### IterativeImputer

1. **Import the necessary libraries:**
    ```python
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    ```

2. **Apply IterativeImputer:**
    ```python
    imputer = IterativeImputer(max_iter=10, random_state=0)
    train[['zero']] = imputer.fit_transform(train[['zero']])
    ```

### Deep Learning Imputer

1. **Import the necessary libraries:**
    ```python
    import pandas as pd
    import numpy as np
    import random
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    ```

2. **Create and prepare data:**
    ```python
    # Sample DataFrame creation
    train = pd.DataFrame({
        'zero': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    # Introduce NaN values
    column = train['zero'].copy()
    missing_pct = int(column.size * 0.4)
    i = [random.choice(range(column.shape[0])) for _ in range(missing_pct)]
    column.iloc[i] = np.NaN
    train['zero'] = column

    # Split the data into features and target
    X = train.drop(columns=['zero'])
    y = train['zero']

    # Replace NaN values with a placeholder (e.g., mean of the column)
    y_placeholder = y.fillna(y.mean())

    # Scale the data
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y_placeholder.values.reshape(-1, 1))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=0)
    ```

3. **Build and train the neural network model:**
    ```python
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2)
    ```

4. **Impute missing values using the trained model:**
    ```python
    y_pred = model.predict(X_test)
    y_pred_original_scale = scaler.inverse_transform(y_pred)
    train_imputed = train.copy()
    train_imputed.loc[y.isna(), 'zero'] = y_pred_original_scale.ravel()
    ```

## License

This project is licensed under the MIT License.
