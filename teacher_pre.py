import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the dataset with ratings
df_ratings = pd.read_csv("teacher_evaluations.csv")

# Select the features and target variable
X = df_ratings[['punctuality', 'arrangement', 'communicationSkills']]  # Features
y_teacher_quality = df_ratings['teachingSubjectMatter']  # Target variable

# Encode the target variable into consecutive integers (Good: 1, Not Good: 0)
le = LabelEncoder()
y_teacher_quality_encoded = le.fit_transform(y_teacher_quality)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_teacher_quality_encoded, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training set
rf_model.fit(X_train, y_train)

# Make predictions on the testing set
predictions = rf_model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model accuracy:", accuracy)

# Define a function to predict whether a teacher is good in the future or not
def predict_teacher_quality(punctuality, arrangement, communication_skills):
    # Create a new DataFrame with the input features
    new_teacher = pd.DataFrame(
        {'punctuality': [punctuality], 'arrangement': [arrangement], 'communicationSkills': [communication_skills]})

    # Use the trained model to make a prediction
    prediction = rf_model.predict(new_teacher)

    # Use the LabelEncoder to transform the predicted label back to the original label
    predicted_label = le.inverse_transform(prediction)

    return predicted_label

# Get the values from the CSV columns
punctuality = df_ratings['punctuality'].values[0]
arrangement = df_ratings['arrangement'].values[0]
communication_skills = df_ratings['communicationSkills'].values[0]

# Make a prediction
result = predict_teacher_quality(punctuality, arrangement, communication_skills)

# Calculate the improvement percentage
initial_quality_encoded = y_teacher_quality_encoded[0]
result_encoded = le.transform([result])[0]
improvement_percentage = ((result_encoded - initial_quality_encoded) / (initial_quality_encoded + 1e-9)) * 100

print("Predicted label (original):", result)
print("Improvement percentage:", improvement_percentage, "%")