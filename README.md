# 📄 SmartDocAI: Classify, Title, Justify

**SmartDocAI** is an intelligent document classification system that allows users to upload PDF, DOCX, or TXT files. It analyzes the content using traditional ML models and LLMs to:

- Classify documents into categories like **Business, Tech, Medical, Finance**, etc.
- Generate an **engaging title** for the document
- Provide **natural language explanations** for both the title and the predicted category

---

## 🚀 Key Features

- Upload `.pdf`, `.docx`, or `.txt` files
- Classify documents using **Logistic Regression**, **SVM**, and **Random Forest**
- Display all predictions with confidence scores
- Highlight the **best prediction**
- Use **LangChain + Together AI (LLaMA-3)** to:
  - Generate a suitable title
  - Explain why the title fits
  - Justify the predicted category
- Let the user:
  - Choose which model’s prediction to use
  - Adjust LLM generation **temperature**
  - **Download** a full PDF report of predictions and explanations

---


### 🔐 Set Your Together.ai API Key 

To securely access your Together.ai API key in Streamlit, create a file named:

```
.streamlit/secrets.toml
```

And inside it, add:

```toml
TOGETHER_API_KEY = "your_together_api_key"
```

---

## 📁 Repository Structure

```
SmartDocAI/
│
├── .streamlit/
│   └── secrets.toml               # Stores Your Together.ai API key 
│
├── app.py                         # Main Streamlit app
├── requirements.txt               # Project dependencies
├── .gitignore                     # Files/folders to exclude from version control
├── .env                           # Optional environment variables (ignored by Git)
│
├── data/                          # Sample and training data
│   ├── bbc-text.csv               # Dataset for training
│   ├── sample_doc.docx            # Sample Word document for testing
│   ├── sample_pdf.pdf             # Sample PDF document for testing
│   └── sample.txt                 # Sample text file for testing
│
├── models/                        # Saved machine learning models
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── svm.pkl
│
├── notebooks/                    
│   ├── train_models.ipynb         # Training 3 models and saving them
│   ├── test_predictions.ipynb     # Test and predict on uploaded files
│   └── tfidf_vectorizer.joblib    # Saved TF-IDF vectorizer
│
├── reports/                      
│   └── classification_results.csv # Summary of model evaluation results
└──
```

---

## 🪪 License

This project is open-source under the MIT License.
