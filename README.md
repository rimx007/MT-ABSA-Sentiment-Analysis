# MT-ABSA: Multi-Task Aspect-Based Sentiment Analysis

**M.S. Thesis Project — Xi'an Technological University, 2026**

A multi-task learning framework for aspect-level sentiment analysis and consumer 
behaviour prediction, using a shared RoBERTa encoder fine-tuned on Amazon product reviews.

## Results

| Model | Macro F1 | Accuracy | AUC |
|---|---|---|---|
| MT-RoBERTa (Ours) | **0.9273** | **0.9380** | **0.9809** |
| Logistic Regression (Best Baseline) | 0.8681 | 0.8812 | 0.9518 |

Statistical validation: 5 seeds · t = 4.06 · **p = 0.0154**

## Architecture
- Shared RoBERTa encoder (roberta-base)
- Task 1: 3-class aspect sentiment classification
- Task 2: Binary recommendation prediction
- SHAP explainability analysis
- Streamlit deployment dashboard

## Tech Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)

## Dataset
- Amazon product reviews corpus (McAuley)
- 100,000 samples from 701,000 original reviews
- 6 product aspects: Quality, Price, Shipping, Durability, Design, Service

## Publication
Hadush R.Y. — *Empowering Diverse Learners: The Role of AI in Inclusive Education.*  
Journal of Education Reform and Innovation, 2024, Vol.2(1): 78–86.

## Conference
Oral Presentation — 8th International Conference on Computer Network, Electronic and Automation  
(ICCNEA 2025), Xi'an, China, September 2025.
