import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

CLASS_NAMES = {0: "Low", 1: "Moderate", 2: "High", 3: "Critical"}

def main():
    print("\n" + "=" * 60)
    print("ARES 2.0 - FEATURE / TARGET ANALYSIS")
    print("=" * 60)

    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))

    print("\n========== DATA ==========")
    print(f"X_train : {X_train.shape}")
    print(f"y_train : {y_train.shape}")
    print(f"X_val   : {X_val.shape}")
    print(f"y_val   : {y_val.shape}")

    print("\n========== MUTUAL INFORMATION ==========")
    discrete_mask = np.array([len(np.unique(X_train[:, i])) <= 10
                              for i in range(X_train.shape[1])])
    mi = mutual_info_classif(
        X_train, y_train,
        discrete_features=discrete_mask,
        random_state=42
    )

    print("\n========== RANDOM FOREST IMPORTANCE ==========")
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    importance = rf.feature_importances_

    rows = []
    for i in range(X_train.shape[1]):
        means = []
        for c in range(4):
            vals = X_train[y_train == c, i]
            means.append(float(np.mean(vals)) if len(vals) else np.nan)

        rows.append({
            "feature": i,
            "low_mean": means[0],
            "moderate_mean": means[1],
            "high_mean": means[2],
            "critical_mean": means[3],
            "class_mean_range": float(np.nanmax(means) - np.nanmin(means)),
            "mutual_information": float(mi[i]),
            "random_forest_importance": float(importance[i])
        })

    df = pd.DataFrame(rows)

    for col in ["mutual_information", "random_forest_importance",
                "class_mean_range"]:
        mx = df[col].max()
        df[col + "_normalized"] = df[col] / mx if mx > 0 else 0.0

    df["combined_signal_score"] = (
        0.40 * df["mutual_information_normalized"] +
        0.40 * df["random_forest_importance_normalized"] +
        0.20 * df["class_mean_range_normalized"]
    )

    df = df.sort_values(
        "combined_signal_score", ascending=False
    ).reset_index(drop=True)

    print("\n========== TOP 20 FEATURES ==========")
    print(df[[
        "feature", "mutual_information",
        "random_forest_importance",
        "class_mean_range",
        "combined_signal_score"
    ]].head(20).to_string(index=False))

    print("\n========== CLASS-WISE TARGET DISTRIBUTION ==========")
    for c in range(4):
        n = int(np.sum(y_train == c))
        print(f"{CLASS_NAMES[c]:<10}: {n:4d} ({n/len(y_train)*100:6.2f}%)")

    print("\n========== LOGISTIC REGRESSION BASELINE ==========")
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xv = scaler.transform(X_val)

    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(Xtr, y_train)

    train_acc = lr.score(Xtr, y_train)
    val_acc = lr.score(Xv, y_val)

    print(f"Train accuracy : {train_acc:.4f}")
    print(f"Val accuracy   : {val_acc:.4f}")

    report_path = os.path.join(DATA_DIR, "feature_target_analysis.csv")
    df.to_csv(report_path, index=False)

    print("\n========== ANALYSIS COMPLETE ==========")
    print(f"Report saved to:\n{report_path}")

    if val_acc < 0.50:
        print("\nWARNING: Current features show weak predictive signal.")
        print("Next step: feature engineering / target-feature review.")
    else:
        print("\nUseful predictive signal detected.")
        print("Next step: model and feature optimization.")

if __name__ == "__main__":
    main()