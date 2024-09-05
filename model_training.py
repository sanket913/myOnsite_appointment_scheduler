
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
from data_preprocessing import load_and_preprocess_data

X_train, X_test, y_train, y_test = load_and_preprocess_data('data/noshowappointments-kagglev2-may-2016.csv')

#Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

#save file
with open('models/schedule_optimizer.pkl', 'wb') as f:
    pickle.dump(model, f)
