const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");

const backendRoot = path.join(__dirname, "..");
const dataRoot = path.join(backendRoot, "data");
const resultsRoot = path.join(backendRoot, "results");
const pythonPath = path.join(backendRoot, ".venv", "Scripts", "python.exe");

const artifactPath = (root, name) => path.join(root, name);

function readCsv(filePath, limit) {
    if (!fs.existsSync(filePath)) {
        return { columns: [], rows: [] };
    }

    const lines = fs.readFileSync(filePath, "utf8")
        .trim()
        .split(/\r?\n/)
        .filter(Boolean);

    if (lines.length === 0) {
        return { columns: [], rows: [] };
    }

    const columns = lines[0].split(",");
    const rows = lines.slice(1, limit + 1).map((line) => (
        line.split(",").map((value) => {
            const number = Number(value);
            return value !== "" && Number.isFinite(number) ? number : value;
        })
    ));

    return { columns, rows };
}

function readMetrics() {
    const metrics = {};
    const reportPath = artifactPath(resultsRoot, "module5_gan_evaluation.txt");

    if (!fs.existsSync(reportPath)) {
        return metrics;
    }

    const report = fs.readFileSync(reportPath, "utf8");
    const metricPattern = /(Accuracy|Precision|Recall|F1 Score)\s*:\s*([\d.]+)%/g;
    let match;

    while ((match = metricPattern.exec(report)) !== null) {
        metrics[match[1].toLowerCase().replace(" ", "_")] = Number(match[2]);
    }

    const rocMatch = report.match(/ROC AUC\s*:\s*([\d.]+)/);
    if (rocMatch) {
        metrics.roc_auc = Number(rocMatch[1]);
    }

    return metrics;
}

function buildDashboard() {
    const syntheticDataPath = artifactPath(dataRoot, "synthetic_disaster_data.csv");
    const historyPath = artifactPath(resultsRoot, "gan_training_history.csv");
    const syntheticData = readCsv(syntheticDataPath, 8);
    const history = readCsv(historyPath, 1);
    const syntheticCount = fs.existsSync(syntheticDataPath)
        ? Math.max(0, fs.readFileSync(syntheticDataPath, "utf8").trim().split(/\r?\n/).length - 1)
        : 0;

    return {
        metrics: readMetrics(),
        synthetic_count: syntheticCount,
        feature_count: syntheticData.columns.length,
        preview: {
            columns: syntheticData.columns.slice(0, 6),
            rows: syntheticData.rows.map((row) => row.slice(0, 6))
        },
        latest_training: history.rows[0]
            ? Object.fromEntries(history.columns.map((column, index) => [column, history.rows[0][index]]))
            : null,
        assets: {
            confusion_matrix: "/results/gan_confusion_matrix.png",
            performance: "/results/gan_overall_performance.png",
            roc_curve: "/results/gan_roc_curve.png",
            training_loss: "/results/gan_training_loss.png"
        }
    };
}

function runPythonScript(scriptName, res, onSuccess) {
    const scriptPath = path.join(backendRoot, "src", scriptName);
    const process = spawn(pythonPath, [scriptPath], { cwd: backendRoot });
    let errorOutput = "";

    process.stderr.on("data", (data) => {
        errorOutput += data.toString();
    });

    process.on("error", (error) => {
        console.error(`Failed to start ${scriptName}:`, error);
        res.status(500).json({ error: "Could not start Module 5 process", details: error.message });
    });

    process.on("close", (code) => {
        if (code !== 0) {
            console.error(`${scriptName} failed:`, errorOutput);
            return res.status(500).json({ error: "Module 5 process failed", details: errorOutput });
        }

        onSuccess();
    });
}

exports.getDashboard = (req, res) => {
    try {
        res.json(buildDashboard());
    } catch (error) {
        console.error("Failed to load Module 5 dashboard:", error);
        res.status(500).json({ error: "Could not load Module 5 dashboard", details: error.message });
    }
};

exports.generateSyntheticData = (req, res) => {
    runPythonScript("generate_synthetic_data.py", res, () => {
        res.json({
            message: "Synthetic disaster data generated successfully",
            dashboard: buildDashboard()
        });
    });
};

exports.getReport = (req, res) => {
    const reportPath = artifactPath(resultsRoot, "ai_disaster_report.txt");
    if (!fs.existsSync(reportPath)) {
        return res.status(404).json({ error: "AI disaster report has not been generated yet" });
    }

    res.json({ report: fs.readFileSync(reportPath, "utf8") });
};

exports.generateReport = (req, res) => {
    runPythonScript("generate_disaster_report.py", res, () => {
        const reportPath = artifactPath(resultsRoot, "ai_disaster_report.txt");
        res.json({ report: fs.readFileSync(reportPath, "utf8") });
    });
};
