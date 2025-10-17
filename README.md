# 🧠 Artificial Intelligence (AI) 

Various Artificial Intelligence (AI) programs for solving problems, searching and pattern finding.

## 🚩 Difficulty Wise

### 1️⃣ Beginner Level

**1. Python Programming for AI**
*   **Topics:** Syntax, data structures (lists, dictionaries), control flow, functions, NumPy (for numerical computing), Pandas (for data manipulation).
*   **Practice Ideas:**
    *   Build a simple calculator or a number guessing game.
    *   Use Pandas to clean and analyze a dataset (e.g., Titanic dataset).
    *   Use NumPy to perform matrix operations and transformations.

**2. Mathematics Fundamentals**
*   **Topics:** Linear Algebra (vectors, matrices, dot products), Calculus (derivatives, gradients), and basic Statistics (mean, median, variance, standard deviation).
*   **Practice Ideas:**
    *   Implement a function to calculate the dot product of two vectors from scratch.
    *   Manually calculate the gradient of a simple function like f(x) = x².

**3. Introduction to Machine Learning**
*   **Topics:** What is ML? Difference between Supervised, Unsupervised, and Reinforcement Learning. The concept of training vs. testing data.
*   **Practice Ideas:**
    *   Use a library like `scikit-learn` to train a simple linear regression model to predict house prices based on square footage.

**4. Core Supervised Learning Algorithms**
*   **Topics:** Linear Regression, Logistic Regression, k-Nearest Neighbors (k-NN), Decision Trees.
*   **Practice Ideas:**
    *   **Linear/Logistic Regression:** Predict student exam scores based on study hours.
    *   **k-NN:** Build a classifier to identify different species of iris flowers.
    *   **Decision Trees:** Classify whether a passenger on the Titanic survived or not.

**5. Core Unsupervised Learning Algorithms**
*   **Topics:** k-Means Clustering, Principal Component Analysis (PCA).
*   **Practice Ideas:**
    *   **k-Means:** Segment customers of a mall based on their spending and demographic data.
    *   **PCA:** Visualize a high-dimensional dataset (like the Iris dataset) in 2D.

**6. Model Evaluation**
*   **Topics:** Train/Test Split, Cross-Validation, Evaluation Metrics (Accuracy, Precision, Recall, F1-Score for classification; Mean Squared Error for regression).
*   **Practice Ideas:**
    *   Train a model and evaluate it using 5-fold cross-validation, reporting multiple metrics.


### 2️⃣ Intermediate Level

**1. Advanced Supervised Learning**
*   **Topics:** Support Vector Machines (SVM), Ensemble Methods (Random Forests, Gradient Boosting Machines like XGBoost, LightGBM).
*   **Practice Ideas:**
    *   Use a Random Forest or XGBoost model to win a Kaggle playground competition.
    *   Compare the performance of SVM with a kernel against a simple logistic regression model on a non-linearly separable dataset.

**2. Introduction to Neural Networks**
*   **Topics:** Perceptron, Multi-Layer Perceptron (MLP), Activation Functions (Sigmoid, Tanh, ReLU), Loss Functions, Backpropagation.
*   **Practice Ideas:**
    *   Build an MLP from scratch using only NumPy to solve the XOR problem.
    *   Use a high-level framework like Keras/TensorFlow or PyTorch to build an MLP for classifying handwritten digits (MNIST dataset).

**3. Introduction to Deep Learning for Computer Vision**
*   **Topics:** Convolutional Neural Networks (CNNs), layers (Conv2D, Pooling, Fully Connected), popular architectures (LeNet, AlexNet).
*   **Practice Ideas:**
    *   Build a CNN to classify images from the CIFAR-10 dataset.
    *   Use Transfer Learning with a pre-trained model (like VGG16 or ResNet50) to classify your own set of images (e.g., cats vs. dogs).

**4. Introduction to Deep Learning for NLP**
*   **Topics:** Text Preprocessing, Word Embeddings (Word2Vec, GloVe), Recurrent Neural Networks (RNNs), LSTMs.
*   **Practice Ideas:**
    *   Train a simple sentiment analysis model (positive/negative review) using an LSTM on a dataset like IMDB reviews.
    *   Use pre-trained GloVe embeddings as the input layer for your model.

**5. Data Engineering for ML**
*   **Topics:** Feature Engineering, Handling Missing Data, Advanced Data Cleaning, Introduction to MLOps (ML + Operations) concepts.
*   **Practice Ideas:**
    *   Participate in a Kaggle competition and focus extensively on feature engineering to improve your model's score.
    *   Build a pipeline that takes raw data, cleans it, and outputs a trained model.


### 3️⃣ Advanced Level

**1. Advanced Deep Learning Architectures**
*   **Topics:** Transformers, Attention Mechanisms, BERT, GPT.
*   **Practice Ideas:**
    *   Fine-tune a pre-trained BERT model for a specific task like question answering or named entity recognition.
    *   Implement a simple Transformer encoder from scratch (e.g., for translation).

**2. Generative AI**
*   **Topics:** Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), Diffusion Models.
*   **Practice Ideas:**
    *   Train a DCGAN to generate realistic-looking faces (using a dataset like CelebA).
    *   Use a pre-trained Stable Diffusion model to generate images from text prompts and experiment with fine-tuning.

**3. Advanced Computer Vision**
*   **Topics:** Object Detection (YOLO, R-CNN), Image Segmentation (U-Net, Mask R-CNN), Image Generation (see above).
*   **Practice Ideas:**
    *   Implement an object detection model to identify and localize different objects in a webcam feed.
    *   Use a U-Net architecture for a medical image segmentation task.

**4. Advanced NLP & LLMs (Large Language Models)**
*   **Topics:** Prompt Engineering, LLM Fine-tuning (LoRA, QLoRA), RAG (Retrieval-Augmented Generation), AI Agent Frameworks.
*   **Practice Ideas:**
    *   Build a RAG system that uses your own documents (e.g., PDFs) to answer questions accurately.
    *   Fine-tune a small LLM (like Gemma or Phi-3) on a specific style of writing or for a specific task.
    *   Create a simple AI agent that can use tools (e.g., perform a web search, run code).

**5. Reinforcement Learning (RL)**
*   **Topics:** Markov Decision Processes (MDPs), Q-Learning, Policy Gradients, Deep Q-Networks (DQN).
*   **Practice Ideas:**
    *   Implement Q-Learning to solve the FrozenLake or CartPole environment from OpenAI Gym.
    *   Train a DQN agent to play a simple Atari game.

**6. MLOps & Deployment**
*   **Topics:** Model Serving (TensorFlow Serving, TorchServe), Containerization (Docker), Orchestration (Kubernetes), CI/CD for ML, Model Monitoring.
*   **Practice Ideas:**
    *   "Dockerize" one of your intermediate models and create a simple API for it using FastAPI or Flask.
    *   Deploy a model on a cloud service like AWS SageMaker, Google Cloud AI Platform, or Azure ML.

**7. Specialized & Emerging Domains**
*   **Topics:**
    *   **Graph Neural Networks (GNNs):** For social network analysis, recommendation systems.
    *   **Multimodal AI:** Models that process both text and images (e.g., CLIP, GPT-4V).
    *   **AI Ethics & Explainable AI (XAI):** Understanding and mitigating bias, interpreting model decisions.
    *   **AI for Science:** Applying AI to problems in biology (AlphaFold), physics, and chemistry.

## 🎯 Topic Wise

### 1. Machine Learning
- Linear Regression
- Logistic Regression
- Decision Trees and Random Forests
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Feature Engineering and Selection
- Model Evaluation and Cross-Validation

### 2. Deep Learning
- Feedforward Neural Networks (ANN)
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN)
- Long Short-Term Memory (LSTM) Networks
- Gated Recurrent Units (GRU)
- Transformers
- Autoencoders
- Attention Mechanisms
- Dropout, Batch Normalization, and Regularization

### 3. Computer Vision
- Image Classification
- Object Detection (YOLO, SSD, Faster R-CNN)
- Image Segmentation (U-Net, Mask R-CNN)
- Image Augmentation and Preprocessing
- Face Recognition
- Optical Character Recognition (OCR)
- Image Generation (GANs, Diffusion Models)

### 4. Natural Language Processing (NLP)
- Tokenization and Text Preprocessing
- Word Embeddings (Word2Vec, GloVe, FastText)
- Sequence-to-Sequence Models
- Sentiment Analysis
- Named Entity Recognition (NER)
- Text Classification
- Transformer Architectures (BERT, GPT, T5)
- Text Summarization
- Machine Translation
- Chatbot Development

### 5. Reinforcement Learning
- Q-Learning
- Deep Q-Networks (DQN)
- Policy Gradient Methods
- Actor-Critic Algorithms
- Proximal Policy Optimization (PPO)
- Monte Carlo Tree Search
- OpenAI Gym Practice Environments
- Multi-Agent Reinforcement Learning

### 6. Generative AI
- Generative Adversarial Networks (GANs)
- Variational Autoencoders (VAEs)
- Diffusion Models
- Text-to-Image Models
- Text Generation using LLM APIs
- Fine-Tuning and Prompt Engineering

### 7. MLOps and Model Deployment
- Model Serialization (Pickle, Joblib)
- Model Serving with FastAPI or Flask
- REST API Integration
- Dockerizing ML Models
- Continuous Training (CT) and Continuous Deployment (CD)
- Experiment Tracking (MLflow, Weights & Biases)
- Model Monitoring and Drift Detection

### 8. Data Engineering for AI
- Data Cleaning and Preprocessing (NumPy, Pandas)
- Feature Pipelines (Scikit-learn Pipelines)
- Handling Imbalanced Datasets
- Dimensionality Reduction (PCA, t-SNE, UMAP)
- Data Versioning (DVC)
- Dataset Augmentation for Vision/NLP

### 9. Advanced Deep Learning Architectures
- Vision Transformers (ViT)
- Graph Neural Networks (GNN)
- Siamese Networks
- Capsule Networks
- Self-Supervised Learning Architectures
- Multimodal Models (CLIP, BLIP)
- Diffusion Transformers (DiTs)

### 10. Optimization and Training Techniques
- Gradient Descent Variants (SGD, Adam, RMSProp)
- Learning Rate Scheduling
- Loss Functions Design
- Regularization (L1, L2, Dropout)
- Early Stopping
- Mixed Precision Training
- Hyperparameter Optimization (Optuna, Ray Tune)

### 11. Evaluation and Benchmarking 
- Confusion Matrix, ROC-AUC, Precision/Recall
- BLEU, ROUGE, and METEOR (NLP metrics)
- Intersection over Union (IoU) for Vision
- F1 and Accuracy Scores
- Model Robustness and Adversarial Testing
- Cross-Validation and Bootstrapping

---

## 🛠️ Tools and Frameworks
- Python (Undisputed main language in AI)
    - https://www.python.org/downloads/
- Python packages:
    ```bash
    pip install -r requirements.txt`
    ```
- Tools and Frameworks
    - TensorFlow
    - PyTorch
    - Scikit-learn
    - Keras
    - OpenCV
    - Hugging Face Transformers
    - ONNX
    - FastAI
    - LangChain (for LLM applications)
    - & more
- SQL (Data is the fuel for AI and SQL is the main language in data)
    - https://www.mysql.com/downloads/
- C/C++ (Performance-critical AI systems)
    - https://visualstudio.microsoft.com/vs/features/cplusplus/
- R (Statistical analysis and research)
    - https://cran.rstudio.com/
- Java (Enterprise AI systems)
    - https://www.java.com/en/
- Julia (New emerging language)
    - https://julialang.org/downloads/
- Prolog (language for simple relational statements)
    - https://www.swi-prolog.org/download/stable 

---

## ✍️ Endnote

<p align="center">⭐ Star this repo if you found it helpful! Thanks for reading.</p>