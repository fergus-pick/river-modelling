# River Flow Exploration
`rivers_path_signature.ipynb` contains a simple exploration of flow data, formulates a binary classification task using univariate temporal data for high flow rates and trains a `1DCNN` and `feedforward` neural network on this predictive task. The models are trained separately on raw data, and on fixed length path signature representions (which summarise temporal information) for comparison. 

We note that the prediction task is fairly contrived for this simple demo, which is why we have benchmarked basic neural network architectures.

The conda env requirements are located in `environment.yaml`.

[European flood discharge data]([url](https://ewds.climate.copernicus.eu/datasets/efas-historical?tab=download)) is used with the following parameters:
  - Variable: `River discharge in the last 6 hours`
  - Model levels: `Surface level`
  - Year: `2020`
  - Month: `January`
  - Day: `all`
  - Time: `00:00, 06:00, 12:00, 18:00`
  - Geographical area: `North: 52°, West: 5°, South: 48°, East: 10°`
  - Data format: `GRIB`
