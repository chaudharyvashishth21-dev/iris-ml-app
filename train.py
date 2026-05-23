import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("🔄 Loading the Iris dataset...")
iris = load_iris()
X, y = iris.data, iris.target

# Split data: 80% to train the model, 20% to test it later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🧠 Training the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a file so our web app can read it instantly
joblib.dump(model, "iris_model.pkl")
print("💾 Success! 'iris_model.pkl' has been created and saved.")