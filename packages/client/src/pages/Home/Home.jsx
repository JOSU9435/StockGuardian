import { useEffect, useState } from "react";
import styles from "./Home.module.scss";
import ApexCharts from "apexcharts";

import { FormControl, InputLabel, Select, MenuItem } from "@mui/material";

const Home = () => {
  const [stock, setStock] = useState("");
  const [stockSymbols, setStockSymbols] = useState([
    "ACC",
    "AMBUJACEM",
    "ADANIENT",
    "NDTV",
    "ADANIPORTS",
    "ADANIPOWER",
    "ADANITRANS",
    "ADANIGREEN",
    "ATGL",
    "AWL",
  ]);
  const [predictions, setPredictions] = useState([]);

  const handleChange = (event) => {
    setStock(event.target.value);
  };

  useEffect(() => {
    (async () => {
      try {
        const response = await fetch(
          import.meta.env.VITE_BACKEND + "predictions",
        );
        const data = await response.json();
        setStockSymbols(Object.keys(data.message));
        setPredictions(data.message);
      } catch (error) {
        console.log(error);
      }
    })();
  }, []);

  useEffect(() => {
    const plotData = predictions[stock]?.map((pred) => {
      return {
        x: pred.timestamp / 1000000,
        y: pred.close,
        fillColor: pred.Ensemble_Anomaly === 1 ? "#9C27B0" : "#FF4560",
      };
    });

    const options = {
      chart: {
        type: "scatter",
      },
      zoom: {
        enabled: true,
      },
      series: [
        {
          name: "Value per unit (₹)",
          data: plotData,
        },
      ],
      xaxis: {
        type: "datetime",
        title: {
          text: "Time",
        },
      },
      yaxis: {
        title: {
          text: "Value per unit (₹)",
        },
      },
    };

    const chart = new ApexCharts(
      document.querySelector(`.${styles.graph}`),
      options,
    );
    chart.render();

    return () => {
      chart.destroy();
    };
  }, [stock, predictions]);

  return (
    <div className={styles.home}>
      <div className={styles.controls}>
        <div className={styles.title}>STOCKGUARDIAN</div>
        <FormControl
          className={styles.form}
          size="small"
          margin="normal"
          color="secondary"
        >
          <InputLabel>Stock</InputLabel>
          <Select value={stock} label="Age" onChange={handleChange}>
            {stockSymbols.map((sym) => (
              <MenuItem value={sym} key={sym}>
                {sym}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </div>
      <div className={styles.graph}></div>
    </div>
  );
};

export default Home;
