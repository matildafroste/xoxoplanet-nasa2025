# xoxoplanet-nasa2025

NASA Space Apps Challenge 2025.

**Challenge**: A World Away: Hunting for Exoplanets with AI.

**Difficulty**: Advanced.

**Tags**: Artificial Intelligence & Machine Learning, Coding, Data Analysis, Data Management, Data Visualization, Extrasolar Objects, Planets & Moons, Software, Space Exploration

**Team**: XOXOplanet.

**Members**: Matilda Froste, Sebastian Froste, Elena Ermakova.

[**Resulting webapplication**](https://xoxoplanet.streamlit.app/)

## Background

This project was developed as part of the **2025 NASA Space Apps Challenge**, under the theme **"A World Away: Hunting for Exoplanets with AI."**  

The challenge focuses on using NASA’s public datasets from missions such as **Kepler**, **K2**, and **TESS** to classify exoplanet candidates. These missions collected light curves — measurements of stellar brightness over time — which can reveal the presence of exoplanets through periodic dips caused by planetary transits.  

Traditionally, astronomers use statistical techniques like the Box Least Squares (BLS) algorithm to identify transits, followed by manual vetting. However, with hundreds of thousands of stars monitored and millions of candidate signals, manual methods become impractical. Machine learning and deep learning approaches provide a scalable way to improve classification accuracy, reduce false positives, and accelerate scientific discovery.  

In this project, we:  
- Explored and cleaned the NASA Kepler Objects of Interest (KOI) dataset.  
- Performed **Exploratory Data Analysis (EDA)** to understand distributions, class balance, and noise patterns.  
- Engineered features from astrophysical parameters and time-series data.  
- Developed a predictive pipeline of four finetuned classifiers; AdaBoost, GradientBoost, XGBoost, RandomForest.
- Evaluated model performance with metrics such as recall, precision, AUC, confusion matrices, and ROC-curves.  
- Built an interactive ([**Streamlit interface**](https://xoxoplanet.streamlit.app/)) to visualize light curves, show predictions, and compare ML vs DL outputs.  

This combined approach highlights the trade-offs between interpretability and performance, demonstrating how AI can help astronomers efficiently distinguish confirmed exoplanets from false positives.

## Methodology

### Pre-processing
- The Kepler (KOI) data was chosen as the main dataset.

- When downloading, two options were available: “Subset,” which included only the checked columns, and “Full,” which included all columns.

- Empty rows and columns were removed, and rows with more than 50% missing data were dropped.

- For the target column koi_disposition, only “CONFIRMED” and “CANDIDATE” entries were kept, while “FALSE POSITIVE” rows were dropped.

- Exploratory data analysis was performed to gain a better understanding of the dataset.

- Columns not useful for training, such as dates and IDs, were removed, along with other non-numerical columns.

### Model development
- The data was split into X (numeric features) and y (the target column koi_disposition).

- X and y were further divided into train, validation, and test sets: the train set was used for training, the validation set for tuning hyperparameters, and the test set for final evaluation.

- Four classification models were trained and compared, all of which are selectable in the interface:

    - Random Forest: an ensemble of decision trees.

    - Gradient Boosting: a method that corrects errors sequentially.

    - XGBoost: an optimized implementation of gradient boosting.

    - AdaBoost: an ensemble method with sequential learners.

### Evaluation
Evaluation on the validation set returned satisfactory metric scores, and the models were integrated into the interface.

| Model             | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------------------|----------|-----------|--------|----------|---------|
| Random Forest     | 0.866    | 0.881     | 0.896  | 0.888    | 0.932   |
| Gradient Boosting | **0.877**| 0.888     | **0.908** | 0.898    | 0.933   |
| XGBoost           | 0.876    | **0.894** | 0.898  | **0.896**| **0.934**|
| AdaBoost          | 0.862    | 0.876     | 0.893  | 0.885    | 0.927   |

### Resulting user interface

The Streamlit app (see link at the top) provides a live, interactive interface where researchers and scientists can instantly input data from a potential exoplanet candidate and receive a classification result. The more variables entered, the more reliable the prediction becomes. Users can also choose which trained model to apply, guided by the performance metrics shown earlier. This not only makes the workflow transparent and less of a “black box,” but also allows direct comparison between models. The app is designed to be intuitive, enabling both experts and newcomers to experiment with real NASA data in a hands-on way.

## Try it it? - Get Started

Follow these steps to set up the project on your own computer.

Make sure you have python installed ([python.org](https://www.python.org/downloads/)). 

### 1. Clone this repository

You need to get a local copy of this repository (the code lives on GitHub, but you’ll want it on your own machine). Either download as ZIP from GitHub and unzip it into a folder of your choice or type
```
git clone https://github.com/matildafroste/xoxoplanet-nasa2025.git
```
when located in your directory of choice. 

### 2. Virtual environment

In the project folder, run:
```
python -m venv venv
```

Activate it by typing:

On Windows:
```
venv\Scripts\activate
```

On Mac/Linux:
```
source venv/bin/activate
```

When activated, your terminal should show (venv) at the beginning of the line.

### 3. Install required libraries

Now install all the libraries we need for the project. They are listed in a file called requirements.txt. Run:
```
pip install -r requirements.txt
```

### 4. Contribute & run the app locally
You should be good to go! Train models on new data, refactor the code, finetune the hyperparameters, or run the app locally using the command 

```
streamlit run app.py
```

Enjoy!