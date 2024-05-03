import { Router } from "express";
import * as Controllers from "../controllers/index.js";

const router = Router({ mergeParams: true });

router.get("/", Controllers.StockPredictions.getStockprediction);

export default router;
