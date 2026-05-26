import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

print("🔄 Loading dataset...")
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🏗️ Creating an Advanced ML Pipeline...")
# A pipeline links data preprocessing (Scaling) directly to the model
pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('classifier', RandomForestClassifier(random_state=42))
])

print("🎛️ Tuning Hyperparameters using Grid Search...")
# Define a grid of settings we want to test to find the best accuracy
param_grid = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [None, 3, 5, 10]
}

# GridSearchCV tests every combination and selects the best one
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test, y_test)

print(f"🥇 Best Settings Found: {grid_search.best_params_}")
print(f"🎯 Enhanced Model Accuracy: {accuracy * 100:.2f}%")

# Save our advanced pipeline file
joblib.dump(best_model, "advanced_iris_pipeline.pkl")
print("💾 Advanced pipeline saved as 'advanced_iris_pipeline.pkl'!")