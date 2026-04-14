from data_preprocessing import load_data
from model import build_model
from sklearn.model_selection import train_test_split

# Load dataset
data, labels = load_data("dataset")

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Build model
model = build_model()

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("cancer_model.h5")

print("Model trained and saved!")