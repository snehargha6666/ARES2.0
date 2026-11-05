const express = require("express");
const router = express.Router();
const multer = require("multer");
const path = require("path");

const upload = multer({
    storage: multer.diskStorage({
        destination: path.join(__dirname, "uploads"),
        filename: (req, file, callback) => {
            const extension = path.extname(file.originalname).toLowerCase();
            callback(null, `${Date.now()}-${Math.random().toString(16).slice(2)}${extension}`);
        }
    })
});
const visionController = require("../controllers/visionController");

router.post(
    "/analyze",
    upload.single("disasterImage"),
    visionController.analyzeDisasterImage
);

module.exports = router;
