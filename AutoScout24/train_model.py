from sklearn.ensemble import RandomForestRegressor
import joblib

# Upload Data
X_train = joblib.load("X_train.pkl")
y_train = joblib.load("y_train.pkl")

# Model define and train
model = RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train, y_train)

#Save model
joblib.dump(model,"model.pkl")