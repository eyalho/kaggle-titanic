import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_anno_example = pd.read_csv("gender_submission.csv")

'''
PassengerId      int64
Survived         int64 # LABEL
Pclass           int64
Name            object
Sex             object
Age            float64
SibSp            int64
Parch            int64
Ticket          object
Fare           float64
Cabin           object
Embarked        object
dtype: object
'''

y = df_train.Survived
X = df_train.drop("Survived", axis=1)  # label
# For baseline: random forest with numeric values only
X.Sex = X.Sex.map({"male": 1, "female": -1})
X.Embarked = X.Embarked.map({"C85": 1, "C123": 2, "B42": 3})
X = X.drop(["Name", "Ticket", "Cabin"], axis=1)
X = X.fillna(0)

# Split dev-train dev-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_predicted = clf.predict(X_test)

print("f1_score macro", f1_score(y_test, y_predicted, average='binary'))
print("accuracy_score", accuracy_score(y_test, y_predicted))

# submission
X = df_test
# For baseline: random forest with numeric values only
X.Sex = X.Sex.map({"male": 1, "female": -1})
X.Embarked = X.Embarked.map({"C85": 1, "C123": 2, "B42": 3})
X = X.drop(["Name", "Ticket", "Cabin"], axis=1)
X = X.fillna(0)
y_submission = clf.predict(X)
df = pd.DataFrame({"PassengerId": X['PassengerId'], "Survived": y_submission}, columns=['PassengerId', 'Survived'])
df.to_csv("1_baseline.csv", index=False)

# kaggle competitions submit -c titanic -f 1_baseline.csv -m "1_baseline"
# On kaggel: 0.77511 score
