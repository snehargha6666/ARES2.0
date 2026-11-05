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

    function renderModule5Dashboard(data) {
        document.getElementById("module5Result").classList.remove("hidden");
        document.getElementById("syntheticCount").textContent = data.synthetic_count ?? "—";
        document.getElementById("featureCount").textContent = data.feature_count ?? "—";
        document.getElementById("ganAccuracy").textContent = data.metrics.accuracy != null
            ? `${data.metrics.accuracy.toFixed(2)}%`
            : "—";
        document.getElementById("ganAuc").textContent = data.metrics.roc_auc != null
            ? data.metrics.roc_auc.toFixed(3)
            : "—";

        const table = document.getElementById("syntheticPreview");
        const head = table.querySelector("thead");
        const body = table.querySelector("tbody");
        head.replaceChildren();
        body.replaceChildren();
        const headerRow = document.createElement("tr");
        data.preview.columns.forEach((column) => {
            const cell = document.createElement("th");
            cell.textContent = column;
            headerRow.appendChild(cell);
        });
        head.appendChild(headerRow);
        data.preview.rows.forEach((row) => {
            const tableRow = document.createElement("tr");
            row.forEach((value) => {
                const cell = document.createElement("td");
                cell.textContent = typeof value === "number" ? value.toFixed(4) : value;
                tableRow.appendChild(cell);
            });
            body.appendChild(tableRow);
        });
        document.getElementById("ganPerformance").src = `${API_URL}${data.assets.performance}`;
        document.getElementById("ganRocCurve").src = `${API_URL}${data.assets.roc_curve}`;
        document.getElementById("ganConfusionMatrix").src = `${API_URL}${data.assets.confusion_matrix}`;
    }

    function showModule5Error(message) {
        const error = document.getElementById("module5Error");
        error.textContent = `Module 5 failed: ${message}`;
        error.classList.remove("hidden");
    }

    async function loadModule5Dashboard() {
        try {
            const response = await fetch(`${API_URL}/api/module5/dashboard`);
            const data = await response.json();
            if (!response.ok || data.error) {
                throw new Error(data.details || data.error || "Module 5 dashboard failed");
            }
            renderModule5Dashboard(data);
        } catch (error) {
            showModule5Error(error.message);
        }
    }

    async function generateSyntheticData() {
        const button = document.getElementById("generateSyntheticButton");
        const loading = document.getElementById("module5Loading");
        document.getElementById("module5Error").classList.add("hidden");
        loading.classList.remove("hidden");
        button.disabled = true;
        try {
            const response = await fetch(`${API_URL}/api/module5/generate`, { method: "POST" });
            const data = await response.json();
            if (!response.ok || data.error) {
                throw new Error(data.details || data.error || "Synthetic data generation failed");
            }
            renderModule5Dashboard(data.dashboard);
        } catch (error) {
            showModule5Error(error.message);
        } finally {
            loading.classList.add("hidden");
            button.disabled = false;
        }
    }

    async function generateAIReport() {
        const button = document.getElementById("generateReportButton");
        const loading = document.getElementById("module5Loading");
        document.getElementById("module5Error").classList.add("hidden");
        loading.classList.remove("hidden");
        button.disabled = true;
        try {
            const response = await fetch(`${API_URL}/api/module5/report`, { method: "POST" });
            const data = await response.json();
            if (!response.ok || data.error) {
                throw new Error(data.details || data.error || "Report generation failed");
            }
            document.getElementById("aiReport").textContent = data.report;
            document.getElementById("reportResult").classList.remove("hidden");
        } catch (error) {
            showModule5Error(error.message);
        } finally {
            loading.classList.add("hidden");
            button.disabled = false;
        }
    }

    loadModule5Dashboard();

    window.generateSyntheticData = generateSyntheticData;
    window.generateAIReport = generateAIReport;

    const module4Response = await fetch(`${API_URL}/api/module4/dashboard`);
    const module4Data = await module4Response.json();
    if (!module4Response.ok || module4Data.error) {
        throw new Error(module4Data.details || module4Data.error || "Module 4 dashboard failed");
    }
    const bestModel = module4Data.best_model || {};
    document.getElementById("module4BestModel").textContent = bestModel.Model || "—";
    document.getElementById("module4Accuracy").textContent = typeof bestModel.Accuracy === "number"
        ? `${(bestModel.Accuracy * 100).toFixed(2)}%`
        : "—";
    document.getElementById("module4Window").textContent = `${module4Data.metadata.window_size || "—"} events`;
    document.getElementById("module4TestSamples").textContent = module4Data.metadata.testing_samples || "—";
    const module4Table = document.getElementById("module4Models");
    const module4Head = module4Table.querySelector("thead");
    const module4Body = module4Table.querySelector("tbody");
    module4Head.replaceChildren();
    module4Body.replaceChildren();
    const module4Columns = ["Model", "Accuracy", "Precision_Weighted", "Recall_Weighted", "F1_Weighted", "ROC_AUC"];
    const module4Header = document.createElement("tr");
    module4Columns.forEach((column) => {
        const cell = document.createElement("th");
        cell.textContent = column.replaceAll("_", " ");
        module4Header.appendChild(cell);
    });
    module4Head.appendChild(module4Header);
    module4Data.models.forEach((model) => {
        const row = document.createElement("tr");
        module4Columns.forEach((column) => {
            const cell = document.createElement("td");
            cell.textContent = typeof model[column] === "number" ? model[column].toFixed(3) : model[column];
            row.appendChild(cell);
        });
        module4Body.appendChild(row);
    });
    const predictionTable = document.getElementById("module4Predictions");
    const predictionHead = predictionTable.querySelector("thead");
    const predictionBody = predictionTable.querySelector("tbody");
    predictionHead.replaceChildren();
    predictionBody.replaceChildren();
    const predictionColumns = [
        "Actual",
        "Predicted",
        "Probability_Low",
        "Probability_Moderate",
        "Probability_High",
        "Probability_Critical"
    ];
    const predictionHeader = document.createElement("tr");
    predictionColumns.forEach((column) => {
        const cell = document.createElement("th");
        cell.textContent = column.replaceAll("_", " ");
        predictionHeader.appendChild(cell);
    });
    predictionHead.appendChild(predictionHeader);
    module4Data.predictions.forEach((prediction) => {
        const row = document.createElement("tr");
        predictionColumns.forEach((column) => {
            const cell = document.createElement("td");
            const value = prediction[column];
            cell.textContent = typeof value === "number" ? value.toFixed(3) : value;
            row.appendChild(cell);
        });
        predictionBody.appendChild(row);
    });
    document.getElementById("module4AccuracyChart").src = `${API_URL}${module4Data.assets.accuracy}`;
    document.getElementById("module4MetricsChart").src = `${API_URL}${module4Data.assets.metrics}`;
    document.getElementById("module4RocChart").src = `${API_URL}${module4Data.assets.roc}`;
    document.getElementById("module4ConfusionChart").src = `${API_URL}${module4Data.assets.confusion}`;
    document.getElementById("module4Loading").classList.add("hidden");
    document.getElementById("module4Result").classList.remove("hidden");

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

async function analyzeVisionImage() {
    const fileInput = document.getElementById("visionImage");
    const button = document.getElementById("visionButton");
    const loading = document.getElementById("visionLoading");
    const result = document.getElementById("visionResult");
    const error = document.getElementById("visionError");
    const file = fileInput.files[0];

    if (!file) {
        error.textContent = "Please choose an image first.";
        error.classList.remove("hidden");
        return;
    }

    async function loadModule5Dashboard() {
        try {
            const response = await fetch(`${API_URL}/api/module5/dashboard`);
            const data = await response.json();
            if (!response.ok || data.error) {
                throw new Error(data.details || data.error || "Module 5 dashboard failed");
            }
            renderModule5Dashboard(data);
        } catch (error) {
            showModule5Error(error.message);
        }
    }

    function renderModule5Dashboard(data) {
        document.getElementById("module5Result").classList.remove("hidden");
        document.getElementById("syntheticCount").textContent = data.synthetic_count ?? "—";
        document.getElementById("featureCount").textContent = data.feature_count ?? "—";
        document.getElementById("ganAccuracy").textContent = data.metrics.accuracy != null
            ? `${data.metrics.accuracy.toFixed(2)}%`
            : "—";
        document.getElementById("ganAuc").textContent = data.metrics.roc_auc != null
            ? data.metrics.roc_auc.toFixed(3)
            : "—";

        const table = document.getElementById("syntheticPreview");
        const head = table.querySelector("thead");
        const body = table.querySelector("tbody");
        head.replaceChildren();
        body.replaceChildren();
        const headerRow = document.createElement("tr");
        data.preview.columns.forEach((column) => {
            const cell = document.createElement("th");
            cell.textContent = column;
            headerRow.appendChild(cell);
        });
        head.appendChild(headerRow);
        data.preview.rows.forEach((row) => {
            const tableRow = document.createElement("tr");
            row.forEach((value) => {
                const cell = document.createElement("td");
                cell.textContent = typeof value === "number" ? value.toFixed(4) : value;
                tableRow.appendChild(cell);
            });
            body.appendChild(tableRow);
        });

        document.getElementById("ganPerformance").src = `${API_URL}${data.assets.performance}`;
        document.getElementById("ganRocCurve").src = `${API_URL}${data.assets.roc_curve}`;
        document.getElementById("ganConfusionMatrix").src = `${API_URL}${data.assets.confusion_matrix}`;
    }

    function showModule5Error(message) {
        const error = document.getElementById("module5Error");
        error.textContent = `Module 5 failed: ${message}`;
        error.classList.remove("hidden");
    }

    async function generateSyntheticData() {
        const button = document.getElementById("generateSyntheticButton");
        const loading = document.getElementById("module5Loading");
        document.getElementById("module5Error").classList.add("hidden");
        loading.classList.remove("hidden");
        button.disabled = true;
        try {
            const response = await fetch(`${API_URL}/api/module5/generate`, { method: "POST" });
            const data = await response.json();
            if (!response.ok || data.error) {
                throw new Error(data.details || data.error || "Synthetic data generation failed");
            }
            renderModule5Dashboard(data.dashboard);
        } catch (error) {
            showModule5Error(error.message);
        } finally {
            loading.classList.add("hidden");
            button.disabled = false;
        }
    }

    async function generateAIReport() {
        const button = document.getElementById("generateReportButton");
        const loading = document.getElementById("module5Loading");
        document.getElementById("module5Error").classList.add("hidden");
        loading.classList.remove("hidden");
        button.disabled = true;
        try {
            const response = await fetch(`${API_URL}/api/module5/report`, { method: "POST" });
            const data = await response.json();
            if (!response.ok || data.error) {
                throw new Error(data.details || data.error || "Report generation failed");
            }
            document.getElementById("aiReport").textContent = data.report;
            document.getElementById("reportResult").classList.remove("hidden");
        } catch (error) {
            showModule5Error(error.message);
        } finally {
            loading.classList.add("hidden");
            button.disabled = false;
        }
    }

    loadModule5Dashboard();

    const formData = new FormData();
    formData.append("disasterImage", file);
    error.classList.add("hidden");
    result.classList.add("hidden");
    loading.classList.remove("hidden");
    button.disabled = true;

    try {
        const response = await fetch(`${API_URL}/api/vision/analyze`, {
            method: "POST",
            body: formData
        });
        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.details || data.error || "Vision analysis failed");
        }

        function renderModule4Table(tableId, columns, rows, numericColumns) {
            const table = document.getElementById(tableId);
            const head = table.querySelector("thead");
            const body = table.querySelector("tbody");
            head.replaceChildren();
            body.replaceChildren();

            const headerRow = document.createElement("tr");
            columns.forEach((column) => {
                const cell = document.createElement("th");
                cell.textContent = column.replaceAll("_", " ");
                headerRow.appendChild(cell);
            });
            head.appendChild(headerRow);

            rows.forEach((row) => {
                const tableRow = document.createElement("tr");
                columns.forEach((column) => {
                    const cell = document.createElement("td");
                    const value = row[column];
                    cell.textContent = numericColumns.includes(column) && typeof value === "number"
                        ? value.toFixed(3)
                        : value ?? "—";
                    tableRow.appendChild(cell);
                });
                body.appendChild(tableRow);
            });
        }

        function renderModule4Dashboard(data) {
            const best = data.best_model || {};
            document.getElementById("module4BestModel").textContent = best.Model || "—";
            document.getElementById("module4Accuracy").textContent = typeof best.Accuracy === "number"
                ? `${(best.Accuracy * 100).toFixed(2)}%`
                : "—";
            document.getElementById("module4Window").textContent = data.metadata.window_size
                ? `${data.metadata.window_size} events`
                : "—";
            document.getElementById("module4TestSamples").textContent = data.metadata.testing_samples ?? "—";

            renderModule4Table(
                "module4Models",
                ["Model", "Accuracy", "Precision_Weighted", "Recall_Weighted", "F1_Weighted", "ROC_AUC"],
                data.models,
                ["Accuracy", "Precision_Weighted", "Recall_Weighted", "F1_Weighted", "ROC_AUC"]
            );
            renderModule4Table(
                "module4Predictions",
                ["Actual", "Predicted", "Probability_Low", "Probability_Moderate", "Probability_High", "Probability_Critical"],
                data.predictions,
                ["Probability_Low", "Probability_Moderate", "Probability_High", "Probability_Critical"]
            );

            document.getElementById("module4AccuracyChart").src = `${API_URL}${data.assets.accuracy}`;
            document.getElementById("module4MetricsChart").src = `${API_URL}${data.assets.metrics}`;
            document.getElementById("module4RocChart").src = `${API_URL}${data.assets.roc}`;
            document.getElementById("module4ConfusionChart").src = `${API_URL}${data.assets.confusion}`;
            document.getElementById("module4Loading").classList.add("hidden");
            document.getElementById("module4Result").classList.remove("hidden");
        }

        async function loadModule4Dashboard() {
            try {
                const response = await fetch(`${API_URL}/api/module4/dashboard`);
                const data = await response.json();
                if (!response.ok || data.error) {
                    throw new Error(data.details || data.error || "Module 4 dashboard failed");
                }
                renderModule4Dashboard(data);
            } catch (error) {
                document.getElementById("module4Loading").classList.add("hidden");
                const errorBox = document.getElementById("module4Error");
                errorBox.textContent = `Module 4 failed: ${error.message}`;
                errorBox.classList.remove("hidden");
            }
        }

        loadModule4Dashboard();

        document.getElementById("visionPreview").src = URL.createObjectURL(file);
        document.getElementById("visionSeverity").textContent =
            data.severity_analysis?.severity || "Unavailable";
        document.getElementById("visionConfidence").textContent =
            data.severity_analysis?.confidence != null
                ? `${(data.severity_analysis.confidence * 100).toFixed(2)}%`
                : "Unavailable";
        document.getElementById("visionSeverityWarning").textContent =
            data.severity_analysis?.warning || "";

        const detectionList = document.getElementById("visionDetections");
        detectionList.replaceChildren();
        const detectionResult = data.object_detection;
        const detectionItems = Array.isArray(detectionResult)
            ? detectionResult
            : detectionResult?.detections;
        if (Array.isArray(detectionItems) && detectionItems.length > 0) {
            detectionItems.forEach((detection) => {
                const item = document.createElement("li");
                item.textContent = `${detection.label} (${(detection.confidence * 100).toFixed(1)}%)`;
                detectionList.appendChild(item);
            });
        } else {
            const item = document.createElement("li");
            item.textContent = detectionResult?.error || "No objects detected above the confidence threshold.";
            detectionList.appendChild(item);
        }

        if (detectionResult?.annotated_image_path) {
            document.getElementById("visionDetectionsImage").src =
                `${API_URL}/${detectionResult.annotated_image_path}`;
        }
        if (detectionResult?.model) {
            document.getElementById("visionDetectionModel").textContent =
                `${detectionResult.model} detections at ${(detectionResult.confidence_threshold * 100).toFixed(0)}% confidence.`;
            if (detectionResult.warning) {
                document.getElementById("visionDetectionModel").textContent +=
                    ` ${detectionResult.warning}`;
            }
        }

        if (data.segmentation?.mask_path) {
            document.getElementById("visionMask").src =
                `${API_URL}/${data.segmentation.mask_path}`;
        }
        document.getElementById("visionSegmentationMethod").textContent = "";
        result.classList.remove("hidden");
    } catch (requestError) {
        error.textContent = `Vision analysis failed: ${requestError.message}`;
        error.classList.remove("hidden");
    } finally {
        loading.classList.add("hidden");
        button.disabled = false;
    }
}
