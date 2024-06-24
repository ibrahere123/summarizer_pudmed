# summarizer_pudmed
 PubMed Article Summarizer Project Report
1. Introduction
The PubMed Article Summarizer project aims to develop a web application that allows users to summarize scientific articles retrieved from PubMed. The application leverages natural language processing (NLP) techniques, specifically using the T5 model from Hugging Face's Transformers library for text summarization.
2. Project Overview
2.1. Objective
The primary objective of the project is to create a user-friendly tool for summarizing PubMed articles into brief or detailed summaries based on user input.
2.2. Key Features
•	Article Summarization: Summarizes PubMed articles based on user-selected summary type (brief or detailed).
•	Input Options: Supports text input or file upload (TXT, MD, PDF) for article retrieval.
•	Interactive Interface: Uses Streamlit for creating an interactive web interface.
•	Model Integration: Integrates a fine-tuned T5 model for generating accurate summaries.
3. Methodology
3.1. Data Exploration
Since PubMed articles are publicly accessible, the project assumes access to a collection of scientific articles for summarization. Sample articles were used for testing and development purposes.
3.2. Model Selection
The T5 model (Text-To-Text Transfer Transformer) was selected for its versatility in NLP tasks, including text summarization. The model's ability to generate human-like summaries makes it suitable for summarizing scientific texts.
3.3. Model Fine-Tuning
The T5 model was fine-tuned using a dataset of PubMed articles and their corresponding summaries. Fine-tuning involves adjusting the model's parameters to optimize its performance specifically for the task of summarizing scientific literature.
3.4. Web Application Development
Technology Stack
•	Framework: Streamlit
•	Language: Python
•	Libraries: Transformers, Streamlit
Application Structure
The web application (app.py) consists of:
•	Main Interface: User input options for article retrieval (text input or file upload).
•	Summarization Logic: Utilizes the fine-tuned T5 model for generating summaries based on user-selected type (brief or detailed).
•	Interactive Sidebar: Provides instructions, input options, and summary type selection.
3.5. Deployment
The application was deployed using Streamlit Sharing, a platform for deploying and sharing Streamlit applications. Deployment steps included:
•	Setting up a Streamlit Sharing account.
•	Connecting the GitHub repository containing the application code (app.py).
•	Configuring deployment settings (Python version, installation commands).
•	Deploying the application to Streamlit Sharing.
4. Results and Discussion
The PubMed Article Summarizer was successfully developed and deployed. Key outcomes include:
•	Functionality: The application accurately summarizes PubMed articles based on user specifications (brief or detailed).
•	User Interface: The Streamlit interface provides a seamless experience for users to interact with the application.
•	Performance: The fine-tuned T5 model demonstrates effective summarization capabilities, enhancing accessibility to scientific knowledge.
5. Conclusion
The PubMed Article Summarizer project demonstrates the integration of advanced NLP techniques into a practical application for scientific article summarization. The use of the T5 model, coupled with Streamlit for web development, facilitates user-friendly access to complex scientific information.
6. Future Work
Future enhancements and considerations for the project include:
•	Expansion of Datasets: Incorporate larger datasets for further model fine-tuning.
•	Enhanced User Interactivity: Introduce features for summarization customization and result visualization.
•	Performance Optimization: Continuously optimize model performance and application responsiveness based on user feedback.
7. References
•	Hugging Face Transformers Documentation: https://huggingface.co/transformers/
•	Streamlit Documentation: https://docs.streamlit.io/
•	PubMed: https://pubmed.ncbi.nlm.nih.gov/


