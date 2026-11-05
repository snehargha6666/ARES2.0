const fs = require("fs");
const path = require("path");

const resultsRoot = path.join(__dirname, "..", "results");

function readCsv(name, limit) {
    const filePath = path.join(resultsRoot, name);
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

function readMetadata() {
    const metadataPath = path.join(resultsRoot, "module4_metadata.json");
    if (!fs.existsSync(metadataPath)) {
        return {};
    }
    return JSON.parse(fs.readFileSync(metadataPath, "utf8"));
}

function toObjects(table) {
    return table.rows.map((row) => (
        Object.fromEntries(table.columns.map((column, index) => [column, row[index]]))
    ));
}

exports.getDashboard = (req, res) => {
    try {
        const metadata = readMetadata();
        const metrics = toObjects(readCsv("module4_metrics.csv", 10));
        const bestModel = metrics.reduce(
            (best, current) => !best || current.Accuracy > best.Accuracy ? current : best,
            null
        );
        const predictions = readCsv("module4_predictions_bilstm.csv", 8);
        const history = readCsv("module4_history_bilstm.csv", 1);

        res.json({
            metadata: {
                window_size: metadata.window_size,
                sequence_method: metadata.sequence_method,
                class_names: metadata.class_names || [],
                training_samples: metadata.training_samples,
                validation_samples: metadata.validation_samples,
                testing_samples: metadata.testing_samples
            },
            models: metrics,
            best_model: bestModel,
            predictions: toObjects(predictions),
            latest_training: toObjects(history)[0] || null,
            assets: {
                accuracy: "/results/module4_accuracy_comparison.png",
                metrics: "/results/module4_precision_recall_f1.png",
                confusion: "/results/module4_confusion_matrices.png",
                roc: "/results/module4_roc_curve.png",
                training: "/results/module4_training_accuracy.png"
            }
        });
    } catch (error) {
        console.error("Failed to load Module 4 dashboard:", error);
        res.status(500).json({ error: "Could not load Module 4 dashboard", details: error.message });
    }
};
