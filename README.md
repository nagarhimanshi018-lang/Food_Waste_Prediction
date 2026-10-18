
# AI-Based Food Demand Prediction and Waste Reduction System

## Project Overview
This project uses machine learning to predict meal consumption based on factors such as day of the week, temperature, rainfall, and special events. It provides an interactive Streamlit dashboard to help food service providers plan meal preparation and reduce potential food surplus.

## Objectives
- Predict meal demand using historical data.
- Analyze the relationship between weather, weekdays, special events, and meal consumption.
- Display historical data and predictions through a dashboard.
- Support better food preparation planning and responsible food consumption.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- Random Forest Regression
- CSV dataset

## Project Files
- `app.py` — Streamlit dashboard and prediction interface.
- `train_model.py` — trains and evaluates the machine learning model.
- `food_demand_model.pkl` — saved trained model.
- `food_demand_sample.csv` — sample dataset used for the project.

## Model Evaluation
The initial model evaluation produced the following results:

- **Mean Absolute Error (MAE):** 9.51 meals
- **Root Mean Squared Error (RMSE):** 11.63 meals
- **R² Score:** 0.554

These results are based on the sample dataset and do not guarantee the same performance on real-world data.

## How to Run the Project

### 1. Install Python
Install Python and ensure it is available from your terminal.

### 2. Install Required Libraries
```bash
pip install streamlit pandas scikit-learn
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```

Make sure the dataset and trained model are stored in the file locations expected by `app.py`.

## Expected Impact
The system aims to support better meal preparation decisions and reduce potential food surplus. Actual food-waste reduction would need to be measured in a real-world trial.

## Future Scope
- Use real-world cafeteria or restaurant data.
- Improve model performance.
- Add database integration.
- Deploy the dashboard online.

## Author
Developed as part of the IBM SkillsBuild ML & Applied AI Internship.

## SDG Alignment
United Nations Sustainable Development Goal 12: Responsible Consumption and Production.
