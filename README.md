## Hate Speech Detection 

This project implements Hate Speech Detection using Logistic Regression and Naive Bayes classifiers built from scratch in Python. The project utilizes Natural Language Processing (NLP) techniques with libraries such as NLTK, WordCloud, and word_tokenize to achieve high accuracy in detecting hate speech.

### Overview

Hate speech detection has become a critical task in social media and online platforms to prevent the spread of harmful content and maintain a safe environment for users. This project aims to build a robust hate speech detection system using machine learning techniques.

### Features

- **End-to-End Project**: The project covers all stages from data preprocessing to model evaluation, creating an end-to-end solution.
  
- **Custom Implementations**: Logistic Regression and Naive Bayes classifiers are implemented from scratch, providing a deeper understanding of the algorithms.

- **NLP Techniques**: NLTK is used for text preprocessing, including tokenization, stemming, and stopword removal.

- **Visualization**: WordCloud is employed to visualize frequent words in hate speech, aiding in understanding the nature of the data.

### Speech Types

### a) Normal Speech
![normal speech](https://github.com/figo2001/Hate-Speech-Deetction/assets/78696850/168f0857-fcfa-4ccb-b59c-5ce660ed513a)
### b) Hated Speech
![Hate Speech](https://github.com/figo2001/Hate-Speech-Deetction/assets/78696850/316bf5d0-d4f9-42c3-bb5f-2a3cfd67d5f9)



### Key Libraries Used

- **NLTK**: For natural language processing tasks such as tokenization and stopwords removal.
  
- **WordCloud**: For generating word clouds to visualize text data.

### Installation

To run this project, you need to have Python installed along with the following libraries:

```bash
pip install nltk
pip install wordcloud
```

### How to Use

1. Clone the repository:

```bash
git clone https://github.com/yourusername/yourrepository.git
```

2. Navigate to the project directory:

```bash
cd yourrepository
```

3. Run the Jupyter Notebook or Python scripts to execute the code:

```bash
jupyter notebook
```

### Project Structure

```
|- data/                   # Dataset files
|  |- hate_speech.csv
|- images/                 # Images used in README
|  |- hate_speech_detection.png
|- notebooks/              # Jupyter notebooks
|  |- Hate_Speech_Detection.ipynb
|- src/                    # Python scripts
|  |- logistic_regression.py
|  |- naive_bayes.py
|- README.md               # Project README
```

### Results

The models achieve high accuracy in detecting hate speech, making it effective for real-world applications. Evaluation metrics such as precision, recall, and F1-score are provided to assess the model performance.

### Future Improvements

- Explore more advanced NLP techniques like word embeddings.
- Enhance model performance by tuning hyperparameters.
- Deploy the model as a web service for real-time hate speech detection.


### Demo 

### Conclusion

Hate Speech Detection is a crucial task in maintaining online platforms' safety and fostering positive interactions. This project demonstrates the effectiveness of Logistic Regression and Naive Bayes classifiers in detecting hate speech with high accuracy. By leveraging NLP techniques and custom implementations, the project provides insights into building robust hate speech detection systems.
