# Campus Energy Optimizer

Campus Energy Optimizer is a machine-learning-based smart building energy management system that predicts energy consumption and recommends an optimal operating action while considering comfort, cost, and carbon emissions.

The project uses real IoT smart-building data from the **University of Sharjah (UoS) M5 Dataset** and provides an interactive **Streamlit dashboard** for visualization and analysis.

## Features

- Real smart-building IoT dataset
- Energy consumption prediction using Ridge Regression
- Temperature, humidity, occupancy, and temporal feature analysis
- Utility-based energy optimization
- Low, Medium, and High operating recommendations
- Interactive Streamlit dashboard
- Configurable comfort, cost, and emission priorities
- Energy consumption and prediction visualizations

## System Workflow

Raw IoT Dataset  
↓  
Data Preprocessing  
↓  
Feature Engineering  
↓  
Ridge Regression Model  
↓  
Energy Prediction  
↓  
Utility-Based Decision Agent  
↓  
Optimal Action Recommendation  
↓  
Streamlit Dashboard

## Dataset

The project uses the **University of Sharjah M5 Smart-Building IoT Dataset**.

**Dataset DOI:** `10.5281/zenodo.12750891`

The dataset contains measurements such as:

- Temperature
- Humidity
- Motion/occupancy information
- Voltage
- Current
- Power
- Energy consumption

Motion-sensor activity is used as an occupancy proxy in the project.

## Machine Learning Model

The energy prediction module uses **Ridge Regression**.

### Input Features

- Temperature
- Humidity
- Occupancy
- Hour
- Day of week
- Month
- Weekend status
- Room type

### Target

`energy`

The dataset is divided into:

- 80% Training
- 20% Testing

Numerical features are standardized and missing values are handled before training.

## Energy Optimization

After predicting energy consumption, the decision agent evaluates three possible operating actions:

- Low
- Medium
- High

The utility function considers:

- User comfort
- Energy cost
- Carbon emissions

The action with the highest expected utility is selected as the recommended operating mode.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- NetworkX

## Project Structure

    campus-energy-optimizer/
    │
    ├── app.py
    ├── agent.py
    ├── ml_model.py
    ├── preprocess_dataset.py
    ├── download_dataset.py
    ├── requirements.txt
    │
    └── data/
        ├── raw/
        └── processed/

## Installation

Clone the repository:

    git clone <your-repository-url>
    cd campus-energy-optimizer

Install dependencies:

    pip install -r requirements.txt

## Run the Project

Download the dataset:

    python download_dataset.py

Preprocess the dataset:

    python preprocess_dataset.py

Run the prediction model:

    python ml_model.py

Generate optimized actions:

    python agent.py

Launch the dashboard:

    python -m streamlit run app.py

## Evaluation

The regression model is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The generated predictions are compared with actual energy consumption to evaluate model performance.

## Future Scope

Future improvements can include:

- Additional benchmark energy datasets
- Random Forest, XGBoost, or LSTM models
- Real-time IoT sensor integration
- Weather API integration
- Dynamic electricity pricing
- Real-time carbon intensity
- Automated HVAC control
- Cloud deployment

## Conclusion
Campus Energy Optimizer demonstrates how machine learning and utility-based decision making can be combined for smart-building energy management. 
The system predicts energy requirements from real IoT measurements and recommends operating actions that balance energy consumption, comfort, cost, and environmental impact.
