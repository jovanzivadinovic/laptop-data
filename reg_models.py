from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

class RegModel:
    def linearRegression(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        reg = LinearRegression()
        reg.fit(X_train_scaled, y_train)
        y_pred = reg.predict(X_test_scaled)
        return mean_squared_error(y_test, y_pred), root_mean_squared_error(y_test, y_pred)
    
    def randomForest(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        reg = RandomForestRegressor()
        reg.fit(X_train_scaled, y_train)
        y_pred = reg.predict(X_test_scaled)
        return mean_squared_error(y_test, y_pred), root_mean_squared_error(y_test, y_pred)
    
    def gradientBoosting(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        reg = GradientBoostingRegressor()
        reg.fit(X_train_scaled, y_train)
        y_pred = reg.predict(X_test_scaled)
        return mean_squared_error(y_test, y_pred), root_mean_squared_error(y_test, y_pred)