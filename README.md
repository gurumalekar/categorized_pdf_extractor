
# PDF Text Categorization and One-Hot Encoding

This project provides a complete pipeline for extracting text from PDF files, categorizing the extracted text into predefined categories based on keywords, and one-hot encoding the categorized data. The final output is saved as a CSV file, making it suitable for further analysis and machine learning tasks.

## Project Overview

This project is designed to work with research papers, specifically focusing on a paper about real-time emotion detection using OpenCV and deep learning. It can be adapted for other topics by adjusting the keyword categories. The main functionalities include:

1. **Extracting Text from PDF Files**: Using `PyMuPDF` (fitz) to read and extract text from PDF files.
2. **Categorizing Extracted Text**: Utilizing `nltk` for sentence tokenization and categorizing each sentence based on the presence of specific keywords.
3. **One-Hot Encoding the Categorized Data**: Applying `scikit-learn`'s `OneHotEncoder` to convert categorical labels into a binary format.
4. **Saving the Processed Data**: Storing the one-hot encoded data into a CSV file for easy access and further processing.

## Features

- **Text Extraction**: Reads and extracts text from each page of a PDF file.
- **Text Categorization**: Categorizes sentences into predefined categories based on keywords.
- **One-Hot Encoding**: Converts the categorical data into a binary matrix.
- **CSV Export**: Saves the processed data into a CSV file.

## Installation

To run this project, you need to have Python installed along with the following libraries:

```bash
pip install PyMuPDF pandas nltk scikit-learn
```

## Usage

1. **Clone the Repository**:

```bash
git clone https://github.com/yourusername/pdf-text-categorization.git
cd pdf-text-categorization
```

2. **Download NLTK Data**:

Make sure to download the necessary NLTK data:

```python
import nltk
nltk.download('punkt')
```

3. **Update the Script**:

Replace the placeholder paths and keywords in the script as needed:

```python
# Paths
pdf_path = 'example.pdf'  # Replace with your PDF file path
csv_path = 'one_hot_encoded_text.csv'  # Replace with desired output CSV file path
```

4. **Run the Script**:

Run the script to process your PDF file:

```bash
python pdf_text_categorization.py
```

## Example

Here's an example of how the script processes a PDF on real-time emotion detection using OpenCV and deep learning. The script extracts sentences, categorizes them into sections like "Introduction/Background", "Methods/Methodology", "Data/Dataset", etc., and one-hot encodes these categories.

## Contributing

Contributions are welcome! If you have any improvements or suggestions, feel free to open an issue or submit a pull request.
