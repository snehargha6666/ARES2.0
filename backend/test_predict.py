import json
import subprocess
import sys

data = {
    "Disaster Subgroup": "Hydrological",
    "Disaster Type": "Flood",
    "Disaster Subtype": "Riverine flood",
    "ISO": "IND",
    "Subregion": "Southern Asia",
    "Magnitude Scale": "Km2",

    "Start Year": 2024,
    "Start Month": 7,
    "Start Day": 10,

    "End Year": 2024,
    "End Month": 7,
    "End Day": 15,

    "CPI": 120,
    "Total Deaths": 20,
    "No. Affected": 5000,
    "No. Injured": 100,
    "Total Damage ('000 US$)": 50000
}

process = subprocess.run(
    [sys.executable, "src/predict.py"],
    input=json.dumps(data),
    text=True,
    capture_output=True
)

print("\n========== ARES 2.0 TEST ==========")

if process.stdout:
    try:
        result = json.loads(process.stdout)

        if "error" in result:
            print("ERROR:", result["error"])
        else:
            print("Severity   :", result["severity"])
            print(
                "Confidence :",
                f"{result['confidence'] * 100:.2f}%"
            )

            print("\nProbabilities:")

            for severity, probability in result["probabilities"].items():
                print(
                    f"{severity:<10}: "
                    f"{probability * 100:.2f}%"
                )

    except json.JSONDecodeError:
        print("Invalid JSON returned:")
        print(process.stdout)

if process.stderr:
    print("\nModel log:")
    print(process.stderr)

print("\n========== TEST COMPLETE ==========")