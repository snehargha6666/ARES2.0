import os
import numpy as np
import pandas as pd

print("\n==========================================")
print("ARES 2.0 - MODULE 5")
print("AI DISASTER REPORT GENERATION")
print("==========================================")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

prediction_file = os.path.join(BASE_DIR, "data", "mlp_predictions.csv")
synthetic_file = os.path.join(BASE_DIR, "data", "synthetic_disaster_data.csv")

report_path = os.path.join(RESULT_DIR, "ai_disaster_report.txt")

severity = "Unknown"

if os.path.exists(prediction_file):

    df = pd.read_csv(prediction_file)

    if "Predicted_Severity" in df.columns:

        values = df["Predicted_Severity"].value_counts()

        severity = values.idxmax()

else:

    severity = "No prediction file found"

synthetic_samples = 0

if os.path.exists(synthetic_file):

    synthetic_samples = len(pd.read_csv(synthetic_file))

report = f"""
====================================
ARES 2.0 - AI DISASTER REPORT
====================================

Predicted Dominant Severity:
{severity}

Synthetic Disaster Records Generated:
{synthetic_samples}

AI-Based Recommendations

Low Risk
- Continue monitoring.

Moderate Risk
- Prepare emergency resources.

High Risk
- Alert local authorities.
- Prepare evacuation plans.

Extreme Risk
- Immediate evacuation.
- Deploy emergency response teams.

====================================
End of Report
====================================
"""

with open(report_path, "w") as f:

    f.write(report)

print(report)
print(f"Report saved to {report_path}")