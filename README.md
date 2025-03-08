# Intellihack_RedHawk_Task01
Machine learning model to predict rain for smart agriculture.

This project aims to predict whether it will rain or not based on historical weather data. The dataset includes features like temperature, humidity, and wind speed. The goal is to help farmers plan irrigation, planting, and harvesting more effectively.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Setup](#setup)
4. [Folder Structure](#folder-structure)
5. [Usage](#usage)
6. [Team Collaboration](#team-collaboration)
7. [Contributing](#contributing)
8. [License](#license)

---

## Project Overview
- **Objective**: Build a machine learning model to predict rain (`rain_or_not`) based on historical weather data.
- **Features**:
  - `avg_temperature`: Average temperature in °C
  - `humidity`: Humidity in percentage
  - `avg_wind_speed`: Average wind speed in km/h
  - `rain_or_not`: Binary label (1 = rain, 0 = no rain)
  - `date`: Date of observation
- **Deliverables**:
  - A Jupyter Notebook with data preprocessing, EDA, model training, and evaluation.
  - A system design for real-time predictions using IoT data.

---

## Dataset
The dataset is stored in the `data/` folder:
- **File**: `weather_data.csv`
- **Rows**: 300+ days of weather observations
- **Columns**: `date`,	`avg_temperature`,	`humidity`,	`avg_wind_speed`,	`rain_or_not`,	`cloud_cover`,	`pressure`

---

## Setup
To set up the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/weather-forecasting-project.git
   cd weather-forecasting-project
   
2. **Install Dependencies:**
   Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
3. **Run the Jupyter Notebook:**
   Launch Jupyter Notebook and open the notebook in the notebooks/ folder:

    ```bash
    jupyter notebook notebooks/intellihack_task01_rain_prediction.ipynb

## Folder Structure

  weather-forecasting-project/
  ├── data/                  # Folder for datasets
  │   └── weather_data.xlsx
  ├── notebooks/             # Folder for Jupyter Notebooks
  │   └── weather_forecasting.ipynb
  ├── scripts/               # Folder for Python scripts (if any)
  ├── README.md              # Project description
  └── requirements.txt       # List of Python dependencies
## Usage
  1. Data Preprocessing:
  
    + Handle missing values, incorrect entries, and formatting inconsistencies.
    + Perform feature engineering (e.g., create new features like temperature range).
  
 2.  Exploratory Data Analysis (EDA):
  
    + Analyze relationships between features and the target variable (rain_or_not).
    + Visualize data distributions and correlations.
  
  3. Model Training and Evaluation:
  
    + Train machine learning models (e.g., Logistic Regression, Random Forest, Gradient Boosting).
    + Evaluate models using metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
  
  4. Model Optimization:
  
    + Perform hyperparameter tuning using GridSearchCV or RandomSearchCV.
    
    + Validate the model using cross-validation.
  
  5. Final Output:
  
    + Generate predictions for the next 21 days.
    + Save the predictions in predictions.csv.

## Team Collaboration
  Git Workflow:
    Create a new branch for your work:
    ```bash
      git checkout -b feature/your-feature-name
    ```
    Commit and push your changes:
    ```bash
      git add .
      git commit -m "Your commit message"
      git push origin feature/your-feature-name
    ```
    Create a Pull Request (PR) on GitHub for review.

## Real-Time Collaboration:

  Use Google Colab for real-time editing of the notebook.

  Share the notebook with your team and grant them Editor access.

## Contributing
We welcome contributions! Here’s how you can contribute:
  
  Fork the repository.
  
  Create a new branch (git checkout -b feature/your-feature-name).
  
  Commit your changes (git commit -m "Add your feature").
  
  Push to the branch (git push origin feature/your-feature-name).
  
  Open a Pull Request.

## License
This project is licensed under the []. See the [LICENSE] file for details.

## Contact
For questions or feedback, please contact:

  Kaveen Amarasekara (Team RedHawk @UCSC)
  Email: kaveenamarasekara2@gmail.com
  GitHub: KaveenAmarasekara
