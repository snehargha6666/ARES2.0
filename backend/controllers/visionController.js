const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

exports.analyzeDisasterImage = (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: "No image uploaded" });
    }

    const imagePath = path.resolve(req.file.path);
    const backendRoot = path.join(__dirname, '..');
    const pythonPath = path.join(backendRoot, '.venv', 'Scripts', 'python.exe');
    const pythonScriptPath = path.join(backendRoot, 'module3_vision', 'vision_pipeline.py');

    const pythonProcess = spawn(pythonPath, [pythonScriptPath, '--image', imagePath], {
        cwd: backendRoot
    });

    let outputData = '';
    let errorData = '';

    pythonProcess.stdout.on('data', (data) => {
        outputData += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        errorData += data.toString();
    });

    pythonProcess.on('close', (code) => {
        fs.unlink(imagePath, (unlinkError) => {
            if (unlinkError) {
                console.error("Failed to remove uploaded vision image:", unlinkError);
            }
        });

        if (code !== 0) {
            console.error(`Python script exited with code ${code}`);
            console.error(`Error details: ${errorData}`);
            return res.status(500).json({ error: "Failed to process image through vision pipeline" });
        }

        try {
            const analysisResult = JSON.parse(outputData);
            return res.status(200).json(analysisResult);
        }

        catch (parseError) {
            console.error("Failed to parse Python output:", outputData);
            return res.status(500).json({ error: "Invalid response format from vision pipeline" });
        }
    });

    pythonProcess.on('error', (error) => {
        console.error("Failed to start vision pipeline:", error);
        return res.status(500).json({
            error: "Could not start vision pipeline",
            details: error.message
        });
    });
};
