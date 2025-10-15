import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------
# 1. Load Data
# ---------------------------
data = pickle.load(open('data_face_features.pickle', mode='rb'))

x = np.array(data['data'])
y = np.array(data['label'])

# هر بردار ۱۲۸ ویژگی داره (از مدل openface)
x = x.reshape(-1, 128)

# ---------------------------
# 2. Split Train / Test
# ---------------------------
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=0
)

# ---------------------------
# 3. Normalize Data
# ---------------------------
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ---------------------------
# 4. Define Models
# ---------------------------
model_logistic = LogisticRegression(max_iter=1000)
model_svm = SVC(kernel='linear', probability=True)
model_rf = RandomForestClassifier(n_estimators=100, random_state=0)

# ---------------------------
# 5. Create Voting Classifier
# ---------------------------
voting = VotingClassifier(
    estimators=[
        ('lr', model_logistic),
        ('svm', model_svm),
        ('rf', model_rf)
    ],
    voting='soft'
)

# ---------------------------
# 6. Train the model
# ---------------------------
voting.fit(x_train, y_train)

# ---------------------------
# 7. Evaluate function
# ---------------------------
def get_report(model, x_train, y_train, x_test, y_test):
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    f1_train = f1_score(y_train, y_pred_train, average='macro')
    f1_test = f1_score(y_test, y_pred_test, average='macro')

    print("Accuracy Train:", acc_train)
    print("Accuracy Test:", acc_test)
    print("F1 Score Train:", f1_train)
    print("F1 Score Test:", f1_test)

# ---------------------------
# 8. Run Report
# ---------------------------
get_report(voting, x_train, y_train, x_test, y_test)
