## END TO END MACHINE LEARNING PROJECT

### Data Ingestion (dataingestion.py):

Ingests data, performs feature engineering, and splits it into train and test sets.
Saves the raw and processed data into CSV files.
Configuration for file paths is organized using a DataIngestionConfig class.

### Data Transformation (datatransformation.py):

Features methods for cleaning and transforming the input data.
Utilizes scikit-learn pipelines for numerical and categorical features.
Handles imputation, scaling, and encoding.
Implements SMOTE for handling class imbalance.
Saves the data transformation object for reuse.

### Model Trainer (modeltrainer.py):

Uses various classifiers for training models.
Evaluates models using F1 score.
Selects the best-performing model based on the F1 score on the test set.
Saves the trained model for later use.

### Predict Pipeline (predictpipeline.py):

Uses a trained model and a preprocessor to predict outcomes for new data.
Stores predictions in a CSV file.
Utilizes a CustomData class for creating a DataFrame from user inputs.

### Flask Application (flask_app.py):

Implements a web application using Flask.
Provides a simple HTML form for users to input data and get predictions.
Integrates with the prediction pipeline to make predictions and update a data collection CSV file.

### Logger (logger.py):

Configures logging for the entire project.
Creates log files with timestamps to track errors and information.

### Exception Handling (exception.py):

Contains custom exception classes for better error handling.
Provides detailed error messages.

### Utilities (utils.py):

Houses utility functions for saving, loading objects, and evaluating models.
Implements GridSearchCV for hyperparameter tuning.

### Setup (setup.py):

Sets up the project for packaging and distribution.
Defines project metadata, such as name, version, and dependencies.

### Requirements (requirements.txt):

Lists project dependencies for easy installation.

