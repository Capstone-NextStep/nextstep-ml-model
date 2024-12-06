# Resume Classifier and Generator

This project is a machine learning pipeline designed to classify resumes into job categories and generate personalized resume descriptions based on templates. It includes text preprocessing, deep learning-based classification, and a simple template-based description generator.

---

## **Features**

- **Resume Classification**: Classifies resumes into predefined job categories (e.g., Data Science, Web Developer).
- **Resume Description Generator**: Provides a job-specific description template for resumes.
- **Interactive Predictions**: Accepts resume texts and predicts their job category in real time.

---

## **Dataset**

The dataset used for this project is the **UpdatedResumeDataSet.csv**, which contains:
- `Resume`: Textual content of resumes.
- `Category`: Job categories such as Data Science, Web Developer, etc.

---

## **Requirements**

### Python Libraries:
- `numpy`: For numerical computations.
- `tensorflow`: For building and training the deep learning model.
- `pandas`: For handling tabular data.
- `matplotlib`: For plotting training history.
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
!pip install numpy tensorflow pandas matplotlib scikit-learn
```
**2. Upload Dataset**
Upload the UpdatedResumeDataSet.csv file to your Colab workspace. Use the following code to load it into your project:
```bash
from google.colab import files
uploaded = files.upload()

import pandas as pd
file_path = list(uploaded.keys())[0]
dataset = pd.read_csv(file_path)
```
**3. Run the Script**
Copy the provided Python script into a Colab notebook cell and execute it step by step.

**4. Save and Download Models**
After training, you can save and download your models and preprocessing tools using the following code:
```bash
    from google.colab import files
    model.save('resume_classifier_model.h5')
    files.download('resume_classifier_model.h5')

    import pickle
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    files.download('tokenizer.pickle')

    with open('label_encoder.pickle', 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    files.download('label_encoder.pickle')
```
**5. Interactive Prediction**
Use the inference pipeline to input resume text and get predictions in Colab:
```bash
sample_resume = """Experienced in Python, R, and data visualization tools. Skilled in machine learning and analytics."""
predicted_category = predict_category(sample_resume, loaded_model, loaded_tokenizer, loaded_label_encoder, max_length)
print("Predicted Category:", predicted_category)
```
---

## **Training Visualization**
The model's training and validation accuracy and loss are visualized using Matplotlib. These plots help in analyzing the model's performance during training.
