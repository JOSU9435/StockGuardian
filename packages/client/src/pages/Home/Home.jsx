import { useState } from "react";
import styles from "./Home.module.scss";
import {
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Card,
  CardContent,
  Typography,
} from "@mui/material";

const Home = () => {
  const [stock, setStock] = useState("");

  const handleChange = (event) => {
    setStock(event.target.value);
  };

  return (
    <div className={styles.home}>
      <div className={styles.controls}>
        <FormControl size="small" margin="normal" color="secondary">
          <InputLabel>Stock</InputLabel>
          <Select value={stock} label="Age" onChange={handleChange}>
            <MenuItem value={10}>Ten</MenuItem>
            <MenuItem value={20}>Twenty</MenuItem>
            <MenuItem value={30}>Thirty</MenuItem>
          </Select>
        </FormControl>

        <Button size="medium" variant="contained" color="secondary">
          Submit
        </Button>
      </div>
      <div className={styles.transactions}>
        <Card sx={{ margin: "1rem", backgroundColor: "#E8EEF7" }}>
          <CardContent>
            <Typography variant="h5" component="h2">
              title
            </Typography>
            <Typography variant="body2" color="textSecondary" component="p">
              description
            </Typography>
          </CardContent>
        </Card>
        <Card sx={{ margin: "1rem" }}>
          <CardContent>
            <Typography variant="h5" component="h2">
              title
            </Typography>
            <Typography variant="body2" color="textSecondary" component="p">
              description
            </Typography>
          </CardContent>
        </Card>
        <Card sx={{ margin: "1rem" }}>
          <CardContent>
            <Typography variant="h5" component="h2">
              title
            </Typography>
            <Typography variant="body2" color="textSecondary" component="p">
              description
            </Typography>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Home;
