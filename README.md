# Saudi Used Cars Price Predictor

This is end-to-end project for Saudi Used Cars Price case.


| Section | Description | File(s) |
| --- | --- | --- |
| 1. Data Preprocessing & Model Training | In this file, we will be able to see how the data is being preprocessed and final model being trained | [main.ipynb](main.ipynb) |
| 2. Model Deployment | Model Deployment to Google Cloud Storage, and serve the model via Vertex AI| [upload.ipynb](upload.ipynb) |
| 3. Inference / Prediction | In this section, we will be able to make inference / prediction using new data | [app.py](app.py) |

## Setup

### Install Dependencies

#### 1. Python

#### 2. Create Python Environment

Create using venv or conda.

```bash
python -m venv .venv
```

Activate the environment

```bash
source .venv/bin/activate
```

or

```bash
conda create -n saudi-used-cars python=3.10
```

#### 3. Install Dependencies

```bash
pip install -r requirement.txt
```

## Run the application

### 1. Data Exploration & Model Training

You can run and explore `main.ipynb` to see how the data is being preprocessed and final model being trained.

### 2. Model Deployment

You can run and explore `upload.ipynb` to see how the model is being deployed to Google Cloud Storage, and serve the model via Vertex AI.

### 3. Inference / Prediction

You can run and explore `app.py` to see how the model is being used to make inference / prediction using new data. This web app assumes the model is already deployed to Vertex AI.

Thus, if you want to modify and change the model, run `upload.ipynb` first to deploy the another model to Vertex AI.

To try it directly, you can launch streamlit app by running:

```bash
streamlit run app.py
```

🚀 Happy coding!