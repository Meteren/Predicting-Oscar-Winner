import pandas as pd  # type: ignore
from sklearn.preprocessing import LabelEncoder #type: ignore
from sklearn.preprocessing import StandardScaler #type: ignore
from sklearn.ensemble import RandomForestClassifier #type: ignore
from sklearn.linear_model import LogisticRegression,LinearRegression #type: ignore
from sklearn.pipeline import Pipeline #type: ignore
from sklearn.compose import ColumnTransformer#type: ignore
from sklearn.impute import SimpleImputer #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.model_selection import KFold, cross_val_score,cross_val_predict #type: ignore
from sklearn.metrics import confusion_matrix #type: ignore

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

oscar_winner_count = df["Winner"].sum()

print(oscar_winner_count)

X = df.drop(columns = ["Winner"])
y = df["Winner"]


preprocessor = ColumnTransformer(transformers=[
    ("num",StandardScaler(),["Movie Time","IMDB Rating","IMDB Votes","Tomatometer Rating","Tomatometer Count","Audience Rating","Audience Count",
                            "Tomatometer Top Critics Count","Tomatometer Fresh Critics Count"]),
    ("cat",SimpleImputer(strategy="most_frequent"),["Directors"])],remainder="passthrough")

 
model_lr = Pipeline(steps=[
    ("preprocessor", preprocessor),
   ("classifier", LogisticRegression(solver="saga", max_iter=10000))
])

model_rf = Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("classifier",RandomForestClassifier())])

   
kf = KFold(n_splits=5, shuffle=True, random_state=42)  

# Evaluate the model
y_pred_rf = cross_val_predict(model_rf, X, y, cv=kf)

lr_score = cross_val_score(model_rf,X,y,cv = kf, scoring="accuracy")

# Check the shape of y_pred_lr and y
print(lr_score)
print(f"Shape of true labels: {y.shape}")
print(f"Shape of predicted labels (Random Forest): {y_pred_rf.shape}")

# Generate confusion matrix for Logistic Regression
cm_rf = confusion_matrix(y, y_pred_rf)
print("Confusion Matrix for Random Forest:")
print(cm_rf)

# Print Scores
#print("Cross-validation scores for each fold:", y_pred_lr)
#print("Mean accuracy:", y_pred_lr.mean())










