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

## 🧠 Models & Training

- **Dataset:** [`bbc-text.csv`](https://www.kaggle.com/datasets/moazeldsokyx/bbc-news)
- **Models trained:**
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
- **Vectorizer:** TF-IDF (Saved with `joblib`)
- **Training Script:** [`train_models.ipynb`](notebooks/train_models.ipynb)
  - Trains all 3 models
  - Saves models and vectorizer using `joblib.dump()`
  - Generates:
    - Accuracy
    - Classification report
    - Confusion matrix (with graphs)

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

- `train_models.ipynb`: Train and evaluate models, save artifacts
- `test_predictions.ipynb`: Test new documents, integrate LLM, generate explanations
- `app.py`: Streamlit interface for end-to-end functionality
- `utils/`: Utility scripts for text cleaning, file reading, etc.
- `models/`: Saved model files and vectorizer
- `static/`: Sample test documents

---

## 💼 Tech Stack

- **Scikit-learn** (ML models, TF-IDF)
- **LangChain** (LLM orchestration)
- **Together AI API** (LLaMA-3 model for natural language generation)
- **Streamlit** (Web app interface)
- **FPDF** (Report generation)
- **Pandas, NumPy, Matplotlib, Seaborn** (Data handling and visualization)

---

## 🔐 API Setup

To use LLM features, you’ll need a Together.ai API key. Store it securely in your environment.

---

## 📌 Future Enhancements

- Add support for more file types (e.g., HTML, JSON)
- Enable summarization of documents
- Support multi-language content
- Deploy the app publicly
- Add user authentication

---

## 🪪 License

This project is open-source under the MIT License.
