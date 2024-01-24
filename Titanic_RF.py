import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import *
from sklearn.model_selection import *
from helpers.data_prep import *
from helpers.eda import *
from helpers.helpers import *

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

df = pd.read_pickle("Datasets/prepared_titanic_df.pkl")
check_df(df)

y = df["SURVIVED"]
X = df.drop(["SURVIVED"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17)

rf_model = RandomForestRegressor(random_state=42).fit(X_train, y_train)

# train hatası
y_pred = rf_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# test hatası
y_pred = rf_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

#######################################
# Model Tuning (Hiperparametre Tuning)
#######################################

rf_params = {"max_depth": [5, 8, 10, 12, 14, None],
             "max_features": [15, 17, 19, 21],
             "n_estimators": [500, 750, 1000],
             "min_samples_split": [4, 5, 6]}

rf_model = RandomForestRegressor(random_state=42)
rf_cv_model = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=1).fit(X_train, y_train)
rf_cv_model.best_params_

#######################################
# Final Model
#######################################

rf_tuned = RandomForestRegressor(**rf_cv_model.best_params_).fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))