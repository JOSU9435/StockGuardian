import { spawn } from "child_process";
import * as Response from "../../globals/response/index.js";
import { readFileSync } from "fs";

const getStockprediction = (_req, res) => {
  const model = spawn("python3", ["model/model.py"]);

  model.stderr.on("data", (err) => {
    console.error(`$stderr ${err}`);
    res.json(Response.Error("model failure"));
  });

  model.on("close", (code) => {
    const data = readFileSync("output.json");
    res.json(new Response.Success(JSON.parse(data)));
  });
};

export { getStockprediction };
