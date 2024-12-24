from numpy import dtype
import pandas as pd  # type: ignore
from sklearn.preprocessing import LabelEncoder #type: ignore
from sklearn.preprocessing import StandardScaler,RobustScaler #type: ignore
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier #type: ignore
from sklearn.linear_model import LogisticRegression,LinearRegression #type: ignore
from sklearn.pipeline import Pipeline #type: ignore
from sklearn.compose import ColumnTransformer#type: ignore
from sklearn.impute import SimpleImputer #type: ignore
from sklearn.model_selection import StratifiedKFold, cross_val_score,cross_val_predict,train_test_split #type: ignore
from sklearn.metrics import confusion_matrix #type: ignore
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns #type: ignore
import matplotlib.pyplot as plt #type: ignore
from pandas.plotting import scatter_matrix
from scipy.stats import shapiro

#functions
def PlotConfusionMatrix(cm, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

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
df = df.drop(columns=["Comdey","Thriller","Western","Sport",
                      "Horror","Film-Noir","Animation","Sci-Fi","Biography","Family","Fantasy"])


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
    ("num",RobustScaler(),["Movie Time","IMDB Rating","IMDB Votes","Tomatometer Rating","Tomatometer Count","Audience Rating","Audience Count",
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

model_random_forest = Pipeline(steps=[
    ("preprocessor", preprocessor),
   ("classifier", RandomForestClassifier())
])

#k-fold 
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  

# Evaluate the grad boosting
y_pred_grad_boosting = cross_val_predict(model_grad_boosting, X, y, cv=kf)
accuracy = GetScore(model_grad_boosting,X,y,kf,"accuracy")
precision = GetScore(model_grad_boosting,X,y,kf,"precision")
recall = GetScore(model_grad_boosting,X,y,kf,"recall")
f1 = GetScore(model_grad_boosting,X,y,kf,"f1")
cm_grad_boosting = ExtractConfusionMatrix(y,y_pred_grad_boosting)

# print grad boosting
print(f"Model accuracy: {accuracy}")
print(f"Model Precision: {precision}")
print(f"Model Recall: {recall}")
print(f"F1 Score: {f1}")
print("Confusion Matrix for GradBoosting:")
print(cm_grad_boosting)
PlotConfusionMatrix(cm_grad_boosting, title="Gradient Boosting Confusion Matrix")

#Generate confusion matrix for Decision tree
y_pred_dt = cross_val_predict(model_decision_tree, X, y, cv=kf)
accuracy = GetScore(model_decision_tree,X,y,kf,"accuracy")
precision = GetScore(model_decision_tree,X,y,kf,"precision")
recall = GetScore(model_decision_tree,X,y,kf,"recall")
f1 = GetScore(model_decision_tree,X,y,kf,"f1")
cm_dt = ExtractConfusionMatrix(y,y_pred_dt)

# Print decision tree
print(f"Model accuracy: {accuracy}")
print(f"Model Precision: {precision}")
print(f"Model Recall: {recall}")
print(f"F1 Score: {f1}")
print("Confusion Matrix for Decision Tree:")
print(cm_dt)
PlotConfusionMatrix(cm_dt, title="Decision Tree Confusion Matrix")

# Generate confusion matrix for Random Forest
y_pred_rf = cross_val_predict(model_random_forest, X, y, cv=kf)
accuracy = GetScore(model_random_forest,X,y,kf,"accuracy")
precision = GetScore(model_random_forest,X,y,kf,"precision")
recall = GetScore(model_random_forest,X,y,kf,"recall")
f1 = GetScore(model_random_forest,X,y,kf,"f1")
cm_rf = ExtractConfusionMatrix(y,y_pred_rf)

#print
print(f"Model accuracy: {accuracy}")
print(f"Model Precision: {precision}")
print(f"Model Recall: {recall}")
print(f"F1 Score: {f1}")
print("Confusion Matrix for Random Forest:")
print(cm_rf)
PlotConfusionMatrix(cm_rf, title="Random Forest Confusion Matrix")


#take input
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_grad_boosting.fit(X_train, y_train)
model_decision_tree.fit(X_train, y_train)
model_random_forest.fit(X_train, y_train)

while(True):
    while(True):
        try:
            movie_index = input("Enter Movie Index:")
            movie_index = int(movie_index)
            break
        except ValueError:
            print("Wront value! Try again.")

    print(f"Test data row count:{X_test.shape[0]}")

    if movie_index >= X_test.shape[0]:
        print("Row count is exceeded. Try again.")
        continue
    if movie_index == -1:
        break

    movie_data = X_test.iloc[movie_index:movie_index+1]  

    prediction_grad_boosting = model_grad_boosting.predict(movie_data)

    prediction_decision_tree = model_decision_tree.predict(movie_data)

    prediction_random_forest = model_random_forest.predict(movie_data)

    print(f"Real value: {'Winner' if y_test.iloc[movie_index] == 1 else 'Not a Winner'}")
    print(f"Gradient Boosting Prediction: {'Winner' if prediction_grad_boosting[0] == 1 else 'Not a Winner'}")
    print(f"Decision Tree Prediction: {'Winner' if prediction_decision_tree[0] == 1 else 'Not a Winner'}")
    print(f"Random Forest Prediction: {'Winner' if prediction_random_forest[0] == 1 else 'Not a Winner'}")


#Regression
sns.regplot(x='IMDB Rating', y='Audience Rating', data=df)
plt.title('Regression Plot between IMDB and Audience Ratings')
plt.show()

#Violin
sns.violinplot(x='Winner', y='IMDB Rating', data=df)
plt.title('Violin Plot of IMDB Ratings by Winners')
plt.show()

#Scatter Matrix
scatter_matrix(df[['IMDB Rating', 'Audience Rating', 'Tomatometer Rating', 'Movie Time']], figsize=(12, 12), diagonal='kde')
plt.show()

#Histogram and Boxplot for all numerical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df[col], kde=True, ax=axes[0])
    axes[0].set_title(f'Histogram of {col}')
    sns.boxplot(x=df[col], ax=axes[1])
    axes[1].set_title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()

# Print grad boosting
print(f"Model accuracy: {accuracy}")
print(f"Model Precision: {precision}")
print(f"Model Recall: {recall}")
print("Confusion Matrix for GradBoosting:")
print(cm_grad_boosting)

#Shapiro-Wilk Test
stat, p = shapiro(df['IMDB Rating'])
print(f'Shapiro-Wilk Test: stat={stat}, p={p}')













