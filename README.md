# Intelligent Road Safety

**UML501: Machine Learning**  
Department of Computer Science & Engineering  
Thapar Institute of Engineering & Technology, Patiala  
Academic Session: 2026–2027 (5th Semester)

**Project Team:**  
- Arnav Gupta (Roll No. 1024170057)  
- Jasmandeep Sharma (Roll No. 1024170269)  

**Project Supervisor:** Mr. Harjot Singh (Lab Instructor, Department of CSE)

## Project Overview

This project presents an end-to-end machine learning and decision-support system analyzing UK Department for Transport STATS19 road collision records from 2021–2023. It addresses severe target class imbalance (1.5% Fatal share) using cost-sensitive learning, discovers regional spatial risk macro-zones through K-Means clustering, models daily collision volumes using polynomial regression, extracts descriptive hazard associations via Apriori rule mining, and deploys an interactive 8-view Streamlit decision-support application.

The analysis is structured for academic evaluation and decision support. Severity estimates represent scenario-based risk likelihoods rather than real-time vehicle warnings.

## Project Contents

| Folder or File | Description |
| --- | --- |
| `notebooks/` | Reproducible data preparation and analysis notebooks (`01` through `05`) |
| `src/` | Shared data access and model artifact helpers (`helpers.py`) |
| `dashboard/` | Full-featured 8-view Streamlit decision-support application (`app.py`, `style.css`) |
| `data/raw/` | Source UK STATS19 CSV files (2021–2023) and official DfT open dataset data guide |
| `data/processed/` | Cleaned 150,000-record analysis sample and data dictionary |
| `outputs/` | Figures (`figures/`), interactive GIS map (`maps/hotspots.html`), and tables (`tables/`) |
| `models/` | Serialized model pipelines (`best_classifier.joblib`, `best_regressor.joblib`) and metadata |
| `report/` | Final project report (`Intelligent_Road_Safety_Report.docx`) |
| `presentation/` | Final project presentation (`Intelligent_Road_Safety_Presentation.pptx`) |
| `run_all.py` | Standalone execution script to run all notebooks in sequence and update outputs |

## Set up

Run these commands from the project folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the analysis

To execute all notebooks and regenerate the saved tables, figures, and models:

```powershell
python run_all.py
```

The notebooks can also be opened in Jupyter and run in numeric order (`01` through `05`) from either the project root or the `notebooks/` folder. The raw STATS19 data and the processed project sample are included in the data folders.

## Run the dashboard

From the project folder, start the application with:

```powershell
python -m streamlit run dashboard/app.py
```

Open the local address printed by Streamlit, usually `http://localhost:8501`.

## Data source and attribution

The source data is published by the UK Department for Transport as STATS19 road safety open data. Consult the included data guide for field definitions and the [DfT road safety statistics collection](https://www.gov.uk/government/collections/road-accidents-and-safety-statistics). The data is made available under the Open Government Licence; include the required attribution when redistributing it.
