async function predictSeverity() {

    const data = {
        "Disaster Subgroup": document.getElementById("subgroup").value,
        "Disaster Type": document.getElementById("type").value,
        "Disaster Subtype": document.getElementById("subtype").value,
        "ISO": document.getElementById("iso").value,
        "Subregion": document.getElementById("subregion").value,
        "Magnitude Scale": document.getElementById("magnitude").value,

        "Start Year": Number(document.getElementById("startYear").value),
        "Start Month": Number(document.getElementById("startMonth").value),
        "Start Day": Number(document.getElementById("startDay").value),

        "End Year": Number(document.getElementById("endYear").value),
        "End Month": Number(document.getElementById("endMonth").value),
        "End Day": Number(document.getElementById("endDay").value),

        "CPI": Number(document.getElementById("cpi").value),
        "Total Deaths": Number(document.getElementById("deaths").value),
        "No. Affected": Number(document.getElementById("affected").value),
        "No. Injured": Number(document.getElementById("injured").value),
        "Total Damage ('000 US$)": Number(
            document.getElementById("damage").value
        )
    };

    const resultBox = document.getElementById("result");

    resultBox.innerHTML = "Analyzing disaster...";

    try {

        const response = await fetch(
            "http://localhost:5000/api/predict",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(data)
            }
        );

        const result = await response.json();

        if (!response.ok) {
            throw new Error(
                result.error || "Prediction failed"
            );
        }

        resultBox.innerHTML = `
            <h2>${result.severity}</h2>

            <p>
                Confidence:
                <strong>
                    ${(result.confidence * 100).toFixed(2)}%
                </strong>
            </p>

            <hr>

            <p>Low:
                ${(result.probabilities.Low * 100).toFixed(2)}%
            </p>

            <p>Moderate:
                ${(result.probabilities.Moderate * 100).toFixed(2)}%
            </p>

            <p>High:
                ${(result.probabilities.High * 100).toFixed(2)}%
            </p>

            <p>Critical:
                ${(result.probabilities.Critical * 100).toFixed(2)}%
            </p>
        `;

    } catch (error) {

        console.error(error);

        resultBox.innerHTML = `
            <p>
                Prediction failed:
                ${error.message}
            </p>
        `;
    }
}