import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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
# X = pd.get_dummies(X) # I have tried dummies (on Sex, Embarked) but didn't improve the f1/acc..

# X = X.fillna(0) # replace fillna with an imputer
# my_imputer = SimpleImputer()
# X = my_imputer.fit_transform(X)
# make new columns indicating what will be imputed
cols_with_missing = (col for col in X.columns if X[col].isnull().any())
for col in cols_with_missing:
    X[col + '_was_missing'] = X[col].isnull()
my_imputer = SimpleImputer()
X = pd.DataFrame(my_imputer.fit_transform(X))

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

# X = X.fillna(0)
y_submission = clf.predict(my_imputer.transform(X))
df = pd.DataFrame({"PassengerId": X['PassengerId'], "Survived": y_submission}, columns=['PassengerId', 'Survived'])
df.to_csv("2_baseline.csv", index=False)

# kaggle competitions submit -c titanic -f %d_baseline.csv -m "%d_baseline"
# On kaggel: 0.77511 score
