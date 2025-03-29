from pyexpat import model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
# import dataset
def import_csv(path):
   
    df = pd.read_csv(path);
    return df;

# dataset generated so pricing is adjusted
def classify_market(price):

    if price < 100 :
        return "Affordable"
    elif price <150:
        return "Mid-range"
    else:
        return "Expensive"

# Prepare data
def Data_Prep():

    path = "globalhousingmarket.csv"
    df = import_csv(path)
    # Select Column that we will Classify
    df["Market_Pricing_Tier"] = df["House Price Index"].apply(classify_market)
    # drop irrelevant columns
    x = df.drop(columns=["Country","Year","Market_Pricing_Tier"])
    # encode to numericals to avoid error 
    y = LabelEncoder().fit_transform(df["Market_Pricing_Tier"])

    # scale features to standardize weights
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # split  data into training dataset and testing dataset  test_size=0.5 meaning 50% of data goes to testing due to dataset being 200 entries 
    x_train,x_test,y_train,y_test = train_test_split(x_scaled, y, test_size=0.5, random_state=42)
    
    return x_train, x_test, y_train , y_test
# Train Support Vector Machine
def SVM_train(x_train,y_train):
    # Create Support vector Machine model using Linear kernel
    model = SVC(kernel="linear")
    # Train the model
    model.fit(x_train,y_train)
    # Dump File for later use 
    joblib.dump(model, "svm_model.joblib")
# Train Naive Bayes
def NB_train(x_train,y_train):
    
    model = GaussianNB()
    # Train the model
    model.fit(x_train,y_train)
    # Dump File for later use 
    joblib.dump(model, "nb_model.joblib")

# Print Report
def Classification_report(model,x_test,y_test):

    
    # make predictions based on the trained model
    y_pred = model.predict(x_test)

    print(f"Test labels count: {len(y_test)}")
    print(f"Predictions count: {len(y_pred)}")
    print("Unique classes in test set:", set(y_test))
    print("Unique classes in predictions:", set(y_pred))
    # calculate accuracy
    acc = accuracy_score(y_test, y_pred)

    # calculate precission
    precision = precision_score(y_test, y_pred,average="macro")
    print("\n--- Classification Report ---")
    print(f"(Accuracy): {acc * 100:.2f}%")
    print(f"(Precision): {precision * 100:.2f}%") 
    print(classification_report(y_test, y_pred))  

# train the models 
def train_model():

    x_train, x_test, y_train , y_test = Data_Prep()
   
    
    SVM_train(x_train, y_train)
    NB_train(x_train, y_train)

    
# prep data and return model to print report
def Report():
    x_train, x_test, y_train , y_test = Data_Prep()
    print("Support Vector Machine")
    svm = joblib.load("svm_model.joblib")
    Classification_report(svm, x_test, y_test)
    print("Naive Bayes")
    
    nb = joblib.load("nb_model.joblib")
    Classification_report(nb, x_test, y_test)

    show_confusion_matrix([svm,nb],x_test,y_test,["SVC","NB"])
# Visualize 
def show_confusion_matrix(models, x_test, y_test, labels):
    fig, axis = plt.subplots(1, len(models), figsize=(10*len(models),5))
    class_names = ['Affordable', 'Mid-range', 'Expensive']
    for i, model in enumerate(models):
        display = ConfusionMatrixDisplay.from_estimator(model,x_test,y_test,ax=axis[i], display_labels=class_names)
        axis[i].set_title(labels[i])

    plt.tight_layout()
    plt.show()

def main():

    
    while True:
     print("Enter Selection")
     print("Train_Model: 0")
     print("Report: 1")
     print("Quit:q")
     selection = input("Select:")
     match selection: 
        case "0":
            train_model()
        case "1":
            Report()
        case "q":
            break
        case _:
            print("Invalid selection")


if __name__ == "__main__":

    main()