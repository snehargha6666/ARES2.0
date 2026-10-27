const API_URL = "http://localhost:5000";

// ============================================================
// CHECK API STATUS
// ============================================================

async function checkAPI() {


const status = document.getElementById("apiStatus");

try {

    const response = await fetch(
        `${API_URL}/api/health`
    );

    if (!response.ok) {
        throw new Error("API unavailable");
    }

    status.textContent = "API Online";

    status.classList.remove("offline");
    status.classList.add("online");

    console.log("ARES API: ONLINE");

} catch (error) {

    status.textContent = "API Offline";

    status.classList.remove("online");
    status.classList.add("offline");

    console.error(
        "API health error:",
        error
    );
}


}

// ============================================================
// PREDICT SEVERITY
// ============================================================

async function predictSeverity() {


console.log(
    "========== ARES PREDICTION START =========="
);


// --------------------------------------------------------
// GET HTML ELEMENTS
// --------------------------------------------------------

const button =
    document.getElementById("predictButton");

const loading =
    document.getElementById("loading");

const resultCard =
    document.getElementById("resultCard");

const errorBox =
    document.getElementById("error");


// --------------------------------------------------------
// CHECK REQUIRED ELEMENTS
// --------------------------------------------------------

if (
    !button ||
    !loading ||
    !resultCard ||
    !errorBox
) {

    console.error(
        "Required HTML element is missing."
    );

    return;
}


// --------------------------------------------------------
// INITIAL UI
// --------------------------------------------------------

resultCard.classList.add("hidden");

errorBox.classList.add("hidden");

loading.classList.remove("hidden");

button.disabled = true;


// ========================================================
// COLLECT INPUT DATA
// ========================================================

const data = {

    "Disaster Subgroup":
        document.getElementById("subgroup").value,

    "Disaster Type":
        document.getElementById("type").value,

    "Disaster Subtype":
        document.getElementById("subtype").value,

    "ISO":
        document.getElementById("iso").value,

    "Subregion":
        document.getElementById("subregion").value,

    "Magnitude Scale":
        document.getElementById("magnitude").value,


    "Start Year":
        Number(
            document.getElementById("startYear").value
        ),

    "Start Month":
        Number(
            document.getElementById("startMonth").value
        ),

    "Start Day":
        Number(
            document.getElementById("startDay").value
        ),


    "End Year":
        Number(
            document.getElementById("endYear").value
        ),

    "End Month":
        Number(
            document.getElementById("endMonth").value
        ),

    "End Day":
        Number(
            document.getElementById("endDay").value
        ),


    "CPI":
        Number(
            document.getElementById("cpi").value
        ),

    "Total Deaths":
        Number(
            document.getElementById("deaths").value
        ),

    "No. Affected":
        Number(
            document.getElementById("affected").value
        ),

    "No. Injured":
        Number(
            document.getElementById("injured").value
        ),

    "Total Damage ('000 US$)":
        Number(
            document.getElementById("damage").value
        )
};


console.log(
    "Sending data:",
    data
);


// ========================================================
// SEND DATA TO NODE API
// ========================================================

try {

    console.log(
        "Sending request to Node API..."
    );


    const response = await fetch(
        `${API_URL}/api/predict`,
        {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(data)
        }
    );


    console.log(
        "API response status:",
        response.status
    );


    // ----------------------------------------------------
    // READ RESPONSE
    // ----------------------------------------------------

    const result =
        await response.json();


    console.log(
        "API result:",
        result
    );


    // ----------------------------------------------------
    // HANDLE API ERROR
    // ----------------------------------------------------

    if (
        !response.ok ||
        result.error
    ) {

        throw new Error(
            result.details ||
            result.error ||
            "Prediction failed"
        );
    }


    // ====================================================
    // DISPLAY RESULT
    // ====================================================

    displayResult(result);


    console.log(
        "========== PREDICTION COMPLETE =========="
    );


} catch (error) {

    console.error(
        "Prediction error:",
        error
    );


    errorBox.textContent =
        "Prediction failed: " +
        error.message;


    errorBox.classList.remove(
        "hidden"
    );


} finally {

    loading.classList.add(
        "hidden"
    );

    button.disabled = false;
}


}

// ============================================================
// DISPLAY RESULT
// ============================================================

function displayResult(result) {


const resultCard =
    document.getElementById(
        "resultCard"
    );


// --------------------------------------------------------
// SEVERITY
// --------------------------------------------------------

document.getElementById(
    "severity"
).textContent =
    result.severity;


// --------------------------------------------------------
// CONFIDENCE
// --------------------------------------------------------

document.getElementById(
    "confidence"
).textContent =
    (
        result.confidence * 100
    ).toFixed(2) + "%";


// ========================================================
// PROBABILITIES
// ========================================================

updateProbability(
    "low",
    result.probabilities.Low
);


updateProbability(
    "moderate",
    result.probabilities.Moderate
);


updateProbability(
    "high",
    result.probabilities.High
);


updateProbability(
    "critical",
    result.probabilities.Critical
);


// --------------------------------------------------------
// SHOW RESULT CARD
// --------------------------------------------------------

resultCard.classList.remove(
    "hidden"
);


}

// ============================================================
// UPDATE PROBABILITY
// ============================================================

function updateProbability(
name,
probability
) {


const percentage =
    probability * 100;


// --------------------------------------------------------
// TEXT
// --------------------------------------------------------

const valueElement =
    document.getElementById(
        `${name}Value`
    );


if (valueElement) {

    valueElement.textContent =
        percentage.toFixed(2) + "%";
}


// --------------------------------------------------------
// BAR
// --------------------------------------------------------

const barElement =
    document.getElementById(
        `${name}Bar`
    );


if (barElement) {

    barElement.style.width =
        percentage + "%";
}


}

// ============================================================
// START API CHECK
// ============================================================

checkAPI();
