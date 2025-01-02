from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


class ClsModel:
    def randomForest(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        cls = RandomForestClassifier()
        cls.fit(X_train_scaled, y_train)
        y_pred = cls.predict(X_test_scaled)
        return classification_report(y_test, y_pred)
    
    def gradientBoosting(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        cls = GradientBoostingClassifier()
        cls.fit(X_train_scaled, y_train)
        y_pred = cls.predict(X_test_scaled)
        return classification_report(y_test, y_pred)
    
    def logisticRegression(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        cls = LogisticRegression()
        cls.fit(X_train_scaled, y_train)
        y_pred = cls.predict(X_test_scaled)
        return classification_report(y_test, y_pred)