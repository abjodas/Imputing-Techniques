
# Deep Learning Imputer for Missing Values

This repository contains code to handle missing values in a dataset using a deep learning-based imputer. The imputer is built using TensorFlow/Keras and leverages a neural network to predict and replace missing values.

## Requirements

To run the code, you need to have the following libraries installed:
- pandas
- numpy
- scikit-learn
- tensorflow

You can install the required libraries using the following command:
```sh
pip install pandas numpy scikit-learn tensorflow
```

## Usage

The provided code demonstrates how to create a sample DataFrame with missing values, prepare the data, build and train a neural network model, and use the model to impute missing values.

### Steps:

1. **Import the necessary libraries and create sample data with missing values:**

    ```python
    import pandas as pd
    import numpy as np
    import random
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

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

    print("Original DataFrame with NaNs:")
    print(train)
    ```

2. **Prepare the data:**

    ```python
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

    print("Training and testing data prepared.")
    ```

3. **Build and train the neural network model:**

    ```python
    # Build the neural network model
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2)

    print("Model training completed.")
    ```

4. **Impute missing values using the trained model:**

    ```python
    # Predict missing values
    y_pred = model.predict(X_test)

    # Inverse transform the scaled predictions to original scale
    y_pred_original_scale = scaler.inverse_transform(y_pred)

    # Replace NaN values in the original data
    train_imputed = train.copy()
    train_imputed.loc[y.isna(), 'zero'] = y_pred_original_scale.ravel()

    print("
DataFrame after deep learning imputer:")
    print(train_imputed)
    ```

## License

This project is licensed under the MIT License.
