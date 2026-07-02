import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

# โหลด Dataset
df = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")

# เลือกเฉพาะ Feature ที่ Extension ดึงได้
FEATURES = [
    "URLLength",
    "IsHTTPS",
    "HasTitle",
    "HasFavicon",
    "NoOfURLRedirect",
    "NoOfPopup",
    "NoOfiFrame",
    "HasExternalFormSubmit",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfExternalRef"
]

X = df[FEATURES]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("=" * 40)
print("Accuracy :", accuracy_score(y_test, pred))
print("=" * 40)
print(classification_report(y_test, pred))

dump(model, "model.pkl")

print("Model saved.")