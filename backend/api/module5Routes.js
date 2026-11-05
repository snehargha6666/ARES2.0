const express = require("express");
const module5Controller = require("../controllers/module5Controller");

const router = express.Router();

router.get("/dashboard", module5Controller.getDashboard);
router.post("/generate", module5Controller.generateSyntheticData);
router.get("/report", module5Controller.getReport);
router.post("/report", module5Controller.generateReport);

module.exports = router;
