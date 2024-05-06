# Deep-Learning-Model-using-Neural-Network
This project implements various machine learning models, including Gaussian Naive Bayes, Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, and a deep learning neural network architecture with hyperparameter tuning, for predicting customer churn using features like age, tenure, balance, and credit score.

In this project, I utilized advanced analytics techniques, including machine learning (ML) and deep learning, to predict customer churn in a banking dataset. The aim was to develop a model that could identify customers who are likely to leave the bank, allowing proactive measures to retain them.

Data Exploration and Preprocessing:

Explored the dataset to understand its structure and characteristics.
Checked for missing values and duplicates, and handled them appropriately.
Conducted exploratory data analysis (EDA) to gain insights into the distribution of features and identify potential relationships between variables.
Performed feature engineering to create new features and transform existing ones to enhance predictive power.
Machine Learning Models:

Trained several machine learning models including Gaussian Naive Bayes, Logistic Regression, K-Nearest Neighbors, Decision Tree, and Random Forest classifiers.
Evaluated model performance using classification metrics such as precision, recall, F1-score, and accuracy.
Tuned hyperparameters for the Decision Tree classifier using Grid Search and a manual brute force approach to optimize model performance.
Utilized Random Forest classifier as it showed the best performance among the traditional machine learning models.
Neural Network Model:

Developed a neural network model using TensorFlow to capture complex patterns in the data.
Tuned hyperparameters including the number of nodes, dropout probability, learning rate, and batch size to optimize model performance.
Evaluated the neural network model's accuracy and calculated the mean squared error (MSE) to assess its predictive performance.
Results:

Among the traditional machine learning models, the Random Forest classifier achieved the highest accuracy of 86%.
The neural network model achieved an accuracy of 79% with a mean squared error of 30,142,534, indicating its effectiveness in predicting churn.
