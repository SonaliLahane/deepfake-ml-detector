from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from feature_engineering import extract_features_from_folder

features, labels = extract_features_from_folder("dataset")
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

print(classification_report(y_test, clf.predict(X_test)))
joblib.dump(clf, "models/deepfake_detector.pkl")
