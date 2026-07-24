import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load Dataset
df = pd.read_csv("Fertilizer Prediction.csv")

# Create Label Encoders
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
fertilizer_encoder = LabelEncoder()

# Encode Categorical Columns
df["Soil Type"] = soil_encoder.fit_transform(df["Soil Type"])
df["Crop Type"] = crop_encoder.fit_transform(df["Crop Type"])
df["Fertilizer Name"] = fertilizer_encoder.fit_transform(df["Fertilizer Name"])

# Features and Target
X = df.drop("Fertilizer Name", axis=1)
y = df["Fertilizer Name"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save Model
joblib.dump(model, "model.pkl")

# Save Encoders
joblib.dump(soil_encoder, "soil_encoder.pkl")
joblib.dump(crop_encoder, "crop_encoder.pkl")
joblib.dump(fertilizer_encoder, "fertilizer_encoder.pkl")

print("✅ Model Saved Successfully")