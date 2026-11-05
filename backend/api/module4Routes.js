const express = require("express");
const module4Controller = require("../controllers/module4Controller");

const router = express.Router();

router.get("/dashboard", module4Controller.getDashboard);

module.exports = router;
