# Predictive Medical Test Outcome Classifier 🧪

![Project Banner](https://raw.githubusercontent.com/medlabtech2013/ml-labtest-classifier/main/banner.png)


This project builds a machine learning model that predicts whether a patient's lab test result is **Normal** or **Abnormal** using features like age, gender, glucose, and cholesterol levels.

---

## 🔍 Features
- Simulates a realistic healthcare dataset
- Uses pandas and NumPy for data creation and handling
- Exploratory data analysis with Seaborn and Matplotlib
- Machine learning with Random Forest Classifier (scikit-learn)
- Evaluates performance (accuracy, precision, recall, F1-score)
- Saves trained model using `joblib`

---

## 🧠 Tech Stack
- Python
- Jupyter Notebook
- pandas, numpy
- scikit-learn
- seaborn, matplotlib

---

## 📁 Project Structure
ml-labtest-classifier/
├── data/
│ └── lab_test_data.csv
├── models/
│ └── lab_test_classifier.pkl
├── notebooks/
│ ├── 01_data_creation.ipynb
│ ├── 02_eda.ipynb
│ └── 03_model_training.ipynb
├── README.md
└── requirements.txt

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ml-labtest-classifier.git
   cd ml-labtest-classifier
   
2. Create and activate a virtual environment:

   python -m venv venv
.\venv\Scripts\activate  # Windows

3. Install required libraries:
    pip install -r requirements.txt

4. Launch Jupyter Notebook:
    jupyter notebook

5. Run notebooks in order:

01_data_creation.ipynb → generate and save dataset

02_eda.ipynb → visualize and explore the data

03_model_training.ipynb → train, evaluate, and save the model

✅ Results
Model Accuracy: ~85%

Exported trained model: models/lab_test_classifier.pkl

📸 Sample Output
Exploratory data charts and a confusion matrix go here.

(You can replace this section with images using markdown:
![Alt Text](images/confusion_matrix.png))



