# Introduction 

# Getting Started

## Prerequisites
- Docker

## Setup Instructions

### 1. Build the Docker Image
docker compose up --build

docker run -it [IMAGE-NAME]

## Running the Code

### 2. Run AutoML and Generate Predictions
Run the main.py script to perform AutoML and generate best models for the objectives:

python main.py

Run the Pygmo.py script to find the Pareto Fronts using NSGA-II:

python Pygmo.py

## Output
The paretofront dataset will be saved to:
data/pfs.csv

## Final Repository Structure
.
├── data/                     # Contains input and output datasets
│   ├── dublin_dwellings.csv  # Input dataset
│   └── pfs.csv               # Predicted output dataset
├── src/
├──     AutoGluonAML.py        # Scripts for AutoML and predictions
├──     LHS.py                 # Script to generate LHS samples
├──     main.py                # Main script
├──     Pygmo.py               # Optimisation using Pygmo
├── .dockerignore 
├── .gitignore  
├── compose.yaml
├── Dockerfile
├── environment.yml         
└── README.md               # This file
