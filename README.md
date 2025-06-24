
# heart_disease_predictor

This app has been built using **Streamlit** and deployed with **Streamlit Community Cloud**.

[Visit the app here](https://assignment-regression.streamlit.app/)  
*(No password required)*

This application predicts whether someone is at risk of **heart disease** based on their health and lifestyle information. The model provides a simple user interface to help visualize risk using machine learning predictions.

---

## Features

- Easy-to-use form powered by Streamlit for collecting user health data
- Real-time prediction of heart disease risk based on 2020 CDC dataset
- Converts categorical variables to dummy variables before prediction
- Deployed to Streamlit Cloud, accessible via any web browser

---

## Dataset

The model was trained on the **Heart Disease – CDC 2020 Cleaned Dataset**. This dataset includes information about:

- BMI
- Smoking and Alcohol Consumption
- Stroke history
- Physical and Mental Health status
- Age Category and Sex
- Sleep time and General Health
- Diabetes status
- Physical activity
- Race and other lifestyle indicators

---

## Technologies Used

- **Streamlit** – To build the frontend web application
- **Scikit-learn** – For training and evaluating the machine learning model
- **Pandas** – For data manipulation and preprocessing
- **Matplotlib** & **Seaborn** – For exploratory analysis and optional visualization

---

## Model

The application uses a trained **Random Forest Classifier**. Key preprocessing steps include:

- Binary encoding (e.g., Yes/No → 1/0)
- One-hot encoding for multi-class categorical variables such as `Race`, `AgeCategory`, `GenHealth`, `Diabetic`
- Feature ordering matched with the trained model

---

## Future Enhancements

- Add SHAP-based explainability for individual predictions
- Visual summary of input features before prediction
- Compare results across different models (e.g., Logistic Regression, XGBoost)
- Allow users to upload batch CSV files for bulk prediction

---

## Installation (for local deployment)

To run the app locally, follow these steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/wang0964/regression_model.git
   cd regression_model
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv env
   source env/bin/activate        # On Windows: env\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application**
   ```bash
   streamlit run streamlit.py
   ```

---

## Dependencies

List of required Python libraries:

```txt
streamlit==1.46.0
pandas==2.3.0
scikit-learn==1.7.0
matplotlib==3.8.0
seaborn==0.12.2
```

---

#### Thank you for using the Heart Disease Predictor!  
This project is used for CST2216 Individual Term Project.
