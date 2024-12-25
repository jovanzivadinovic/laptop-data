from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class ClsModel:
    def randomForest(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        cls = RandomForestClassifier()
        cls.fit(X_train, y_train)
        y_pred = cls.predict(X_test)
        return classification_report(y_test, y_pred)
    
    def gradientBoosting(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        cls = GradientBoostingClassifier()
        cls.fit(X_train, y_train)
        y_pred = cls.predict(X_test)
        return classification_report(y_test, y_pred)
    
    def logisticRegression(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        cls = LogisticRegression()
        cls.fit(X_train, y_train)
        y_pred = cls.predict(X_test)
        return classification_report(y_test, y_pred)