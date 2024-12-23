import pandas as pd  # type: ignore
from sklearn.preprocessing import LabelEncoder #type: ignore
from sklearn.preprocessing import StandardScaler #type: ignore
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier #type: ignore
from sklearn.linear_model import LogisticRegression,LinearRegression #type: ignore
from sklearn.pipeline import Pipeline #type: ignore
from sklearn.compose import ColumnTransformer#type: ignore
from sklearn.impute import SimpleImputer #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.model_selection import StratifiedKFold, cross_val_score,cross_val_predict,GridSearchCV #type: ignore
from sklearn.metrics import confusion_matrix #type: ignore
from sklearn.tree import DecisionTreeClassifier
import numpy as np

#functions
def ExtractConfusionMatrix(y,y_pred):
    c_matrix = confusion_matrix(y,y_pred)
    return c_matrix

def GetScore(model,X,y,kf,type):
    score = cross_val_score(model,X,y,cv = kf, scoring=type)
    return score.mean()

def EncodeLabel(column,df):
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    return df

def Discriminate(column,df):
    discriminated_val = pd.get_dummies(df[column])
    discriminated_val = discriminated_val.astype(int)
    df = df.join(discriminated_val)
    df = df.drop(columns=[column])
    return df

#read data
df = pd.read_csv("oscars_df.csv")

#drop unnecessary columns
df = df.drop(columns=["Movie Info","Genres","Critic Consensus","Authors","Actors","Original Release Date",
                      "Streaming Release Date","Film ID","Film","Oscar Year", "Unnamed: 0","Film Studio/Producer(s)","Production Company"])

#fill empty columns
df["Directors"] = df["Directors"].fillna(df["Directors"].mode()[0])
df["Audience Rating"] = df["Audience Rating"].fillna(round(df["Audience Rating"].mean()))
df["Audience Status"] = df["Audience Status"].fillna(df["Audience Status"].mode()[0])
df["Audience Count"] = df["Audience Count"].ffill()
df["Tomatometer Status"] = df["Tomatometer Status"].fillna(df["Tomatometer Status"].mode()[0])
df["Tomatometer Rating"] = df['Tomatometer Rating'].fillna(round(df["Tomatometer Rating"].mean()))
df["Content Rating"] = df["Content Rating"].fillna(df["Content Rating"].mode()[0])
df["Tomatometer Count"] = df["Tomatometer Count"].bfill()
df["Tomatometer Count"] = df["Tomatometer Count"].ffill()
df["Tomatometer Top Critics Count"] = df["Tomatometer Top Critics Count"].bfill()
df["Tomatometer Top Critics Count"] = df["Tomatometer Top Critics Count"].ffill()
df["Tomatometer Fresh Critics Count"] = df["Tomatometer Fresh Critics Count"].bfill()
df["Tomatometer Fresh Critics Count"] = df["Tomatometer Fresh Critics Count"].ffill()
df["Tomatometer Rotten Critics Count"] = df["Tomatometer Rotten Critics Count"].ffill()

#genre discrimination
df_genres = df["Movie Genre"].str.get_dummies(sep = ",")
df = df.join(df_genres)
df = df.drop(columns=["Movie Genre"])
df = df.drop(columns=["Comdey"])

#rotten status discrimination
df = Discriminate("Tomatometer Status",df)

#audience status discrimination
df = Discriminate("Audience Status",df) 

#content rating discrimination
df = Discriminate("Content Rating",df)

#oscar - nominee discrimination
df = Discriminate("Award",df)

#nominee handling
nominees = df["Nominee"].tolist()

for index, i in enumerate(nominees):
    if i == 0:
        nominees[index] = 1

df["Nominee"] = nominees

df = df.drop(columns=["Nominee"])

#type casting
df["Audience Rating"] = df["Audience Rating"].astype(int)
df['IMDB Votes'] = df['IMDB Votes'].str.replace(',', '').astype(int)

#director labeling
df = EncodeLabel("Directors",df)

# Remove some of the movies for model stability
non_winners_indices = df[df["Winner"] == 0].index
num_to_remove = int(len(non_winners_indices) * 0.5)
indices_to_drop = non_winners_indices.to_series().sample(n=230, random_state=42)
df = df.drop(index=indices_to_drop)
df = df.reset_index(drop=True)

#extract to csv
df.to_csv("new_oscars_data.csv",index = False)

#choose columns
X = df.drop(columns = ["Winner"])
y = df["Winner"]

#preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("num",StandardScaler(),["Movie Time","IMDB Rating","IMDB Votes","Tomatometer Rating","Tomatometer Count","Audience Rating","Audience Count",
                            "Tomatometer Top Critics Count","Tomatometer Fresh Critics Count"]),
    ("cat",SimpleImputer(strategy="most_frequent"),["Directors"])],remainder="passthrough")

#Gradient Boosting Model 
model_grad_boosting = Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("classifier",GradientBoostingClassifier())])

#Logistic Regression Model
model_decision_tree = Pipeline(steps=[
    ("preprocessor", preprocessor),
   ("classifier", DecisionTreeClassifier())
])

#k-fold 
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  

# Evaluate the grad boosting
y_pred_grad_boosting = cross_val_predict(model_grad_boosting, X, y, cv=kf)
accuracy = GetScore(model_grad_boosting,X,y,kf,"accuracy")
precision = GetScore(model_grad_boosting,X,y,kf,"precision")
recall = GetScore(model_grad_boosting,X,y,kf,"recall")
cm_grad_boosting = ExtractConfusionMatrix(y,y_pred_grad_boosting)

# print grad boosting
print(f"Model accuracy: {accuracy}")
print(f"Model Precision: {precision}")
print(f"Model Recall: {recall}")
print("Confusion Matrix for GradBoosting:")
print(cm_grad_boosting)

#Generate confusion matrix for Logistic Regression
y_pred_dt = cross_val_predict(model_decision_tree, X, y, cv=kf)
accuracy = GetScore(model_decision_tree,X,y,kf,"accuracy")
precision = GetScore(model_decision_tree,X,y,kf,"precision")
recall = GetScore(model_decision_tree,X,y,kf,"recall")
cm_dt = ExtractConfusionMatrix(y,y_pred_dt)

# Print decision tree
print(f"Model accuracy: {accuracy}")
print(f"Model Precision: {precision}")
print(f"Model Recall: {recall}")
print("Confusion Matrix for Logistic Regression:")
print(cm_dt)












