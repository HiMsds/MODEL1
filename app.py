import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset
file_path = "dm.csv"  # Ensure dm.csv is in the same directory
df = pd.read_csv(file_path, encoding="latin1")

# Drop unnecessary columns
df = df.drop(columns=["Sr. No.", "Date", "Geo_Location", "Place_of_offence", "Key_location"])

# Handle missing values
df = df.dropna()

# Encode categorical variables
label_encoders = {}
for col in ["Police Station", "Head", "Crowd_Density", "Crime_Location_Type"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for later use

# Define features and target variable
X = df.drop(columns=["Crime_Location_Type"])  # Features
y = df["Crime_Location_Type"]  # Target variable

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Save models
with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

with open("dt_model.pkl", "wb") as f:
    pickle.dump(dt_model, f)

# Create Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form
    model_type = data.get("model")  # Choose between "rf" and "dt"

    # Convert input data into a DataFrame
    input_data = {
        "Police Station": label_encoders["Police Station"].transform([data["police_station"]])[0],
        "Year": int(data["year"]),
        "Head": label_encoders["Head"].transform([data["crime_head"]])[0],
        "Week_no.": int(data["week_no"]),
        "Crowd_Density": label_encoders["Crowd_Density"].transform([data["crowd_density"]])[0],
    }
    
    input_df = pd.DataFrame([input_data])

    # Load the chosen model
    model = rf_model if model_type == "rf" else dt_model

    # Get probabilities for each crime location type
    probabilities = model.predict_proba(input_df)[0]
    top_3_indices = np.argsort(probabilities)[-3:][::-1]  # Get top 3 most probable indices

    # Convert predictions back to labels
    top_3_predictions = label_encoders["Crime_Location_Type"].inverse_transform(top_3_indices)
    top_3_probabilities = [probabilities[i] for i in top_3_indices]

    # Format response
    response = [{"crime_location": loc, "probability": round(prob * 100, 2)} for loc, prob in zip(top_3_predictions, top_3_probabilities)]

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
