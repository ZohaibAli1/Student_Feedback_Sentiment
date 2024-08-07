import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset with ratings
df_ratings = pd.read_csv("feedback_data_with_subjective_scores.csv")

# Select the features and target variable
X = df_ratings[['Learning Outcomes']]  # Features
y_teacher_quality = df_ratings['BestFeatures_Subjective_Score']  # Target variable

# Encode the target variable into consecutive integers
le = LabelEncoder()
y_teacher_quality = le.fit_transform(y_teacher_quality)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_teacher_quality, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Assume you have a new course with the following features
new_course = pd.DataFrame({'Learning Outcomes': [4.5]})

# Use the trained model to make a prediction
prediction = rf_model.predict(new_course)

# Print the predicted label (0 or 1)
print("Predicted label:", prediction)

# Use the LabelEncoder to transform the predicted label back to the original label
predicted_label = le.inverse_transform(prediction)
print("Predicted label (original):", predicted_label)