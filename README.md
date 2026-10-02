# Loan Approval Prediction 🌳
A Python machine learning project that uses a Decision Tree to predict loan approval from applicant information


Can a machine learning model make a simple prediction about whether a loan application should be approved?

That is what I explored in this project.

The program uses a small dataset containing information about applicants, including age, income, credit score, and marital status. The `loan_approved` column is used as the target that the model tries to predict.

Before training, the text values for marital status and loan approval are converted into numerical values so that the machine learning model can work with them.

I then split the dataset into training and testing data and trained a Decision Tree classifier. After training, the model makes predictions on the test set, and the accuracy is calculated to see how often those predictions match the known results.

The project also includes a small test with a new applicant. After entering a few example values, the trained model predicts whether the loan would be approved according to the patterns it learned from the dataset.

### The Main Idea

The model receives a few pieces of information about an applicant:

`Age`
`Income`
`Credit Score`
`Married`

and uses them to predict:

`Loan Approved`

The overall flow looks like this:

```text
Loan Dataset
     ↓
Prepare the Data
     ↓
Convert Text to Numbers
     ↓
Train / Test Split
     ↓
Decision Tree
     ↓
Predictions
     ↓
Accuracy
     ↓
Test a New Applicant
```

### Why a Decision Tree?

A Decision Tree is an interesting model for a small classification project because its predictions are based on a series of decisions. In this project, it learns patterns in the applicant data and uses those patterns to classify new examples.

### Technologies

* Python
* Pandas
* Scikit-learn
* CSV Dataset

### Running the Project

Install the required libraries:

```bash
pip install pandas scikit-learn
```

Then run:

```bash
python loan_prediction.py
```

Make sure `loan_prediction.csv` is in the same folder as the Python file.

### What I Learned

This project gave me practice with a complete beginner-friendly classification workflow: loading a dataset, preparing categorical values, selecting features, splitting data, training a Decision Tree, checking accuracy, and using the trained model on new input.

The most useful part was seeing how a model can take several simple pieces of information and turn them into a classification result.

### Note

This project is for learning and demonstrating machine learning concepts. It is not intended to make real financial or lending decisions.
