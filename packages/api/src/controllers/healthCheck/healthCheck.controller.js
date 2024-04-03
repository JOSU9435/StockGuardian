import * as Response from "../../globals/response/index.js";

const healthCheck = (_req, res) => {
  res.json(new Response.Success("OK"));
};

export { healthCheck };
