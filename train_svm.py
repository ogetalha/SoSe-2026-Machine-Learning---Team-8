import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def main():
    features_path = "data/features.npy"
    labels_path = "data/labels.npy"
    model_output_dir = "line_following_solution"
    os.makedirs(model_output_dir, exist_ok=True)
    model_path = os.path.join(model_output_dir, "svm_model.pkl")

    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print("Error: Feature or label files not found! Please run collect_data.py first.")
        return

    X = np.load(features_path)
    y_raw = np.load(labels_path)

    # Discretize targets into 3 classes:
    # -1: Left (steering < -0.10)
    #  0: Straight (-0.10 <= steering <= 0.10)
    #  1: Right (steering > 0.10)
    y = np.zeros_like(y_raw, dtype=np.int32)
    y[y_raw < -0.10] = -1
    y[y_raw > 0.10] = 1

    # Print class counts
    classes, counts = np.unique(y, return_counts=True)
    print("Class counts in full dataset:")
    for c, cnt in zip(classes, counts):
        label = "Left" if c == -1 else "Straight" if c == 0 else "Right"
        print(f"  {label} ({c}): {cnt}")

    # Split train/test (stratified to ensure balanced splits)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nTraining SVM on {len(X_train)} samples, testing on {len(X_test)} samples...")

    # SVC with RBF kernel, balanced class weights, and probability calibration
    clf = SVC(
        C=5.0,
        gamma=0.05,
        kernel='rbf',
        class_weight='balanced',
        probability=True,
        random_state=42
    )

    # Train
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Split Accuracy: {acc * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Left", "Straight", "Right"]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Re-train on the full dataset to get the absolute best representation
    print("\nFitting final model on the full dataset...")
    final_clf = SVC(
        C=5.0,
        gamma=0.05,
        kernel='rbf',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
    final_clf.fit(X, y)

    # Save the final model
    with open(model_path, "wb") as f:
        pickle.dump(final_clf, f)

    print(f"Saved trained SVM model to: {model_path}")

if __name__ == "__main__":
    main()
