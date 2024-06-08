import fitz  # PyMuPDF
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.preprocessing import OneHotEncoder

# Download NLTK data
nltk.download('punkt')

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def categorize_text(text):
    """Categorizes text based on the presence of specific keywords."""
    sentences = sent_tokenize(text)
    categorized_data = []
    
    # Keywords and categories
    keywords_categories = {
        "Introduction/Background": ["introduction", "background", "related work", "previous research", "prior studies", "literature review"],
        "Methods/Methodology": ["method", "methodology", "approach", "technique", "procedure", "framework", "algorithm", "implementation"],
        "Data/Dataset": ["data", "dataset", "data collection", "data preparation", "data preprocessing", "data augmentation", "training data", "validation data", "test data"],
        "Experiments/Experimentation": ["experiment", "experimentation", "experimental setup", "experimental results", "testing", "evaluation", "evaluation metrics"],
        "Results/Findings": ["results", "findings", "outcome", "performance", "accuracy", "precision", "recall", "f1-score", "confusion matrix"],
        "Discussion/Analysis": ["discussion", "analysis", "interpretation", "insights", "observations", "analysis of results", "comparison"],
        "Conclusion/Future Work": ["conclusion", "summary", "future work", "future directions", "limitations", "concluding remarks", "recommendations"],
        "Tools/Technologies": ["OpenCV", "deep learning", "neural networks", "convolutional neural network", "CNN", "TensorFlow", "Keras", "PyTorch", "Haar Cascades"]
    }

    for sentence in sentences:
        categorized = False
        for category, keywords in keywords_categories.items():
            if any(keyword.lower() in sentence.lower() for keyword in keywords):
                categorized_data.append((sentence, category))
                categorized = True
                break
        if not categorized:
            categorized_data.append((sentence, "Uncategorized"))
    
    return categorized_data

def one_hot_encode_data(categorized_data):
    """One-hot encodes the categories of the categorized data."""
    df = pd.DataFrame(categorized_data, columns=['Text', 'Category'])
    
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(df[['Category']])
    
    categories_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(['Category']))
    
    one_hot_encoded_df = pd.concat([df[['Text']], categories_df], axis=1)
    
    return one_hot_encoded_df

def save_to_csv(data, csv_path):
    """Saves data to a CSV file."""
    data.to_csv(csv_path, index=False)

# Paths
pdf_path = 'test.pdf'  
csv_path = 'one_hot_encoded_text1.csv'  

# Process
text = extract_text_from_pdf(pdf_path)
categorized_data = categorize_text(text)
one_hot_encoded_data = one_hot_encode_data(categorized_data)
save_to_csv(one_hot_encoded_data, csv_path)

print(f"One-hot encoded data saved to {csv_path}")
