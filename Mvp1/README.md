# Job Skills Analysis

This project analyzes job opportunities data, focusing on required skills and job titles. It includes data cleaning, text analysis, and visualizations like word clouds to explore trends in job requirements.

---

## **Features**

- **Skill Analysis**: Analyzes the most in-demand skills across different job titles.
- **Visualization**: Generates word clouds to visualize the frequency of skills and other key data points.

---

## **Dataset**

The dataset used for this project is the **Job opportunities.csv**, which contains:

- `Required Skills:`: Skills needed for various job roles.
- `Job Title`: Titles of job opportunities.

---

## **Requirements**

### Python Libraries:

- `numpy`: For numerical computations.
- `pandas`: For handling tabular data.
- `matplotlib`: For plotting training history.
- `wordcloud`: For generating word clouds.
- `tensorflow`: For future machine learning extensions.
- `scikit-learn`: For train-test splitting and label encoding.

### Install Dependencies:

Use the following command to install required libraries:

```bash
pip install -r requirements.txt
```

---

## **Using in Google Colab**

If you are running this project on Google Colab, follow these steps:

**1. Setup Environment**
Google Colab provides most of the required libraries pre-installed, but you may need to install additional libraries using the following command:

```bash
!pip install wordcloud
```

**2. Upload Dataset**
Upload the Job opportunities.csv file to your Colab workspace. Use the following code to load it into your project:

```bash
from google.colab import files
uploaded = files.upload()

import pandas as pd
file_path = list(uploaded.keys())[0]
data = pd.read_csv(file_path)
```

**3. Run the Script**
Copy the provided Python script into a Colab notebook cell and execute it step by step.

**4. Save and Download Models**
After training, you can save and download your models and preprocessing tools using the following code:

```bash
    from google.colab import files
    import tensorflow as tf
    model.save('model_nextStep.h5')
    files.download('model_nextStep.h5')

    tokenizer_json = tokenizer.to_json()
    with open('tokenizer.json', 'w') as f:
        f.write(tokenizer_json)
    files.download('tokenizer.json')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open('nextStep.tflite', 'wb') as f:
    f.write(tflite_model)
    files.download('nextStep.tflite')
```

## **Training Visualization**

The model's training and validation accuracy and loss are visualized using Matplotlib. These plots help in analyzing the model's performance during training, such as identifying overfitting or underfitting trends.
