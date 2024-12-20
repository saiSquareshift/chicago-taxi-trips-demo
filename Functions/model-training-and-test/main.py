import functions_framework
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
from google.cloud import bigquery
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from google.cloud import storage
import joblib

client = bigquery.Client()




def ann(X_train, X_test, y_train, y_test):
    X_train_scaled = X_train
    X_test_scaled = X_test

    
    # Define the model
    model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)  # Output layer
    ])


    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(
    X_train_scaled, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    verbose=1
    )

    # save model
    joblib.dump(model, "ann_model.pkl")
    # Initialize the GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket("chicago-taxi-trip-dataset-01")
    blob = bucket.blob("models/ann_model.pkl")
    
    # Upload the file
    blob.upload_from_filename("ann_model.pkl")

    # Evaluate the model:

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # For the sequential model only:
    y_pred_flat = y_pred.flatten()
    y_test = y_test.values if isinstance(y_test, (pd.DataFrame, pd.Series)) else y_test
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    preds = y_pred.tolist()

    response = {
        "Mean Squared Error": str(mse),
        "Mean Absolute Error": str(mae),
        "Root Mean Squared Error": str(rmse),
        "R-squared Score": str(r2),
        "predictions": preds
    }

    return response


def svr(X_train, X_test, y_train, y_test):
    X_train_scaled = X_train
    X_test_scaled = X_test


    # Create and train the SVR model
    svr = SVR(C=1.0, epsilon=0.1, kernel='rbf')  # These values can be adjusted
    svr.fit(X_train_scaled, y_train)

    # save model
    joblib.dump(model, "svr_model.pkl")
    # Initialize the GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket("chicago-taxi-trip-dataset-01")
    blob = bucket.blob("models/svr_model.pkl")
    
    # Upload the file
    blob.upload_from_filename("svr_model.pkl")

    # Make predictions
    y_pred = svr.predict(X_test_scaled)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    preds = y_pred.tolist()

    response = {
        "Mean Squared Error": str(mse),
        "Mean Absolute Error": str(mae),
        "Root Mean Squared Error": str(rmse),
        "R-squared Score": str(r2),
        "predictions": preds
    }

    return response

def gradient_boosting(X_train, X_test, y_train, y_test,n_est=100, lr=0.1, md=3, rs=42):
    X_train_scaled = X_train
    X_test_scaled = X_test

    # Create and train the Gradient Boosting model
    gb_model = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr, max_depth=md, random_state=rs)
    gb_model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = gb_model.predict(X_test_scaled)

    # save model
    joblib.dump(model, "gradboost_model.pkl")
    # Initialize the GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket("chicago-taxi-trip-dataset-01")
    blob = bucket.blob("models/gradboost_model.pkl")
    
    # Upload the file
    blob.upload_from_filename("gradboost_model.pkl")

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    preds = y_pred.tolist()

    response = {
        "Mean Squared Error": str(mse),
        "Mean Absolute Error": str(mae),
        "Root Mean Squared Error": str(rmse),
        "R-squared Score": str(r2),
        "predictions": preds
    }

    return response




def polynomial_regression(X_train, X_test, y_train, y_test):
    X_train_scaled = X_train
    X_test_scaled = X_test
    def poly_regression(degree):
        # Create a pipeline that scales, creates polynomial features, then fits a linear regression
        model = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree, include_bias=False),
            LinearRegression()
        )

        # Fit the model
        model.fit(X_train_scaled, y_train)
        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_mse = -cv_scores.mean()

        return model, y_pred, cv_mse

    # Try different degrees of polynomial features
    degrees = [1, 2, 3, 4]
    results = []

    for degree in degrees:
        model, y_pred, cv_mse = poly_regression(degree)
        results.append({
            'Degree': degree,
            'CV MSE': cv_mse
        })
        print(f"Degree {degree}:")
        print(f"  CV MSE: {cv_mse:.4f}")
        print()

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Select the best model (you can change the criterion as needed)
    best_degree = results_df.loc[results_df['CV MSE'].idxmin(), 'Degree']
    print(f"Best degree based on CV MSE: {best_degree}")

    # Fit the best model
    best_model, y_pred, best_cv_mse = poly_regression(best_degree)

    # save model
    joblib.dump(model, "polyrgr_model.pkl")
    # Initialize the GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket("chicago-taxi-trip-dataset-01")
    blob = bucket.blob("models/polyrgr_model.pkl")
    
    # Upload the file
    blob.upload_from_filename("polyrgr_model.pkl")

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    preds = y_pred.tolist()

    response = {
        "Mean Squared Error":str(mse),
        "Mean Absolute Error":str(mae),
        "Root Mean Squared Error":str(rmse),
        "R-squared Score":str(r2),
        "predictions": preds
    }

    return response


@functions_framework.http
def hello_http(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """

    # Set CORS headers for the preflight request
    if request.method == "OPTIONS":
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "3600",
        }

        return ("", 204, headers)

    # Set CORS headers for the main request
    headers = {"Access-Control-Allow-Origin": "*"}

    
    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'project_id' in request_json:
        project_id = request_json['project_id']
    elif request_args and 'project_id' in request_args:
        project_id = request_args['project_id']
    else:
        project_id = "gke-elastic-394012"
    
    if request_json and 'dataset_id' in request_json:
        dataset_id = request_json['dataset_id']
    elif request_args and 'dataset_id' in request_args:
        dataset_id = request_args['dataset_id']
    else:
        dataset_id = "chicago_taxi_trips"

    if request_json and 'table_name' in request_json:
        table_name = request_json['table_name']
    elif request_args and 'table_name' in request_args:
        table_name = request_args['table_name']
    else:
        table_name = "sample_taxi_trips"

    if request_json and 'model' in request_json:
        model = request_json['model']
    elif request_args and 'model' in request_args:
        model = request_args['model']
    else:
        model = "poly_rgr"

    
    # Perform a query and get training data
    QUERY = (f"""SELECT PC1, PC2, PC3, PC4, tips FROM `{project_id}.{dataset_id}.{table_name}`""")
    df = client.query_and_wait(QUERY).to_dataframe()
    X_train, X_test, y_train, y_test = train_test_split(df[['PC1', 'PC2', 'PC3', 'PC4']], df['tips'], test_size=0.2, random_state=42)

    try:
        if(model == 'poly_rgr'):
            response = polynomial_regression(X_train, X_test, y_train, y_test)
        elif(model == 'grad_boost'):
            response = gradient_boosting(X_train, X_test, y_train, y_test)
        elif(model == 'svr'):
            response = svr(X_train, X_test, y_train, y_test)
        else:
            response = ann(X_train, X_test, y_train, y_test)
        return (response, 200, headers)
    except Exception as e:
        return (f"Failed job: {str(e)}", 500, headers)
