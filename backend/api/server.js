const express = require("express");
const cors = require("cors");
const { spawn } = require("child_process");
const path = require("path");
const visionRoutes = require("./visionRoutes");
const module5Routes = require("./module5Routes");
const module4Routes = require("./module4Routes");

const app = express();
const PORT = 5000;

app.use(cors());
app.use(express.json());
app.use("/api/vision", visionRoutes);
app.use("/api/module5", module5Routes);
app.use("/api/module4", module4Routes);
app.use("/public", express.static(path.join(__dirname, "..", "public")));
app.use("/results", express.static(path.join(__dirname, "..", "results")));


// ============================================================
// PATHS
// ============================================================

const PYTHON = path.join(
    __dirname,
    "..",
    ".venv",
    "Scripts",
    "python.exe"
);

const predictPath = path.join(
    __dirname,
    "..",
    "src",
    "predict.py"
);


// ============================================================
// HEALTH CHECK
// ============================================================

app.get("/api/health", (req, res) => {
    res.json({
        status: "OK",
        message: "ARES 2.0 API is running"
    });
});


// ============================================================
// PREDICTION API
// ============================================================

app.post("/api/predict", (req, res) => {

    console.log("Prediction request received");

    const python = spawn(
        PYTHON,
        [predictPath]
    );

    let output = "";
    let errorOutput = "";

    python.stdin.write(
        JSON.stringify(req.body)
    );

    python.stdin.end();


    python.stdout.on("data", (data) => {
        output += data.toString();
    });


    python.stderr.on("data", (data) => {
        errorOutput += data.toString();
    });


    python.on("error", (error) => {

        console.error("Failed to start Python:", error);

        res.status(500).json({
            error: "Could not start Python",
            details: error.message
        });
    });


    python.on("close", (code) => {

        console.log("Python exited with code:", code);

        if (code !== 0) {

            console.error(
                "Python error:",
                errorOutput
            );

            return res.status(500).json({
                error: "Prediction failed",
                details: errorOutput
            });
        }


        try {

            const result = JSON.parse(
                output.trim()
            );

            res.json(result);

        } catch (error) {

            console.error(
                "Invalid Python response:",
                output
            );

            res.status(500).json({
                error: "Invalid prediction response",
                details: output
            });
        }

    });

});


// ============================================================
// START SERVER
// ============================================================

const server = app.listen(
    PORT,
    "127.0.0.1",
    () => {

        console.log("");
        console.log("==========================================");
        console.log("ARES 2.0 API SERVER");
        console.log("==========================================");
        console.log(
            `API:      http://localhost:${PORT}`
        );
        console.log(
            `Python:   ${PYTHON}`
        );
        console.log(
            `Predictor:${predictPath}`
        );
        console.log("==========================================");
        console.log("Server is running...");
        console.log("");
    }
);


// ============================================================
// ERROR HANDLING
// ============================================================

server.on("error", (error) => {

    console.error("SERVER ERROR:");

    if (error.code === "EADDRINUSE") {
        console.error(
            `Port ${PORT} is already in use.`
        );
    } else {
        console.error(error);
    }

});