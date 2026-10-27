import requests

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

response = requests.post(
    "http://localhost:5000/api/predict",
    json=data
)

print("\n========== NODE API TEST ==========")
print("Status:", response.status_code)
print("Result:")

try:
    print(response.json())
except:
    print(response.text)

print("===================================")