import pandas as pd  # type: ignore
from sklearn.preprocessing import LabelEncoder #type: ignore

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
                      "Streaming Release Date","Film ID"])

#fill empty columns
df["Directors"] = df["Directors"].fillna(df["Directors"].mode()[0])
df["Production Company"] = df["Production Company"].fillna("Unknown")
df["Audience Rating"] = df["Audience Rating"].fillna(round(df["Audience Rating"].mean()))
df["Audience Status"] = df["Audience Status"].fillna(df["Audience Status"].mode()[0])
df["Audience Count"] = df["Audience Count"].ffill()
df["Tomatometer Status"] = df["Tomatometer Status"].fillna(df["Tomatometer Status"].mode()[0])
df["Tomatometer Rating"] = df['Tomatometer Rating'].fillna(round(df["Tomatometer Rating"].mean()))
df["Content Rating"] = df["Content Rating"].fillna(df["Content Rating"].mode()[0])
df["Tomatometer Count"] = df["Tomatometer Count"].bfill()
df["Tomatometer Top Critics Count"] = df["Tomatometer Top Critics Count"].bfill()
df["Tomatometer Fresh Critics Count"] = df["Tomatometer Fresh Critics Count"].bfill()
df["Tomatometer Rotten Critics Count"] = df["Tomatometer Rotten Critics Count"].ffill()


#genre discrimination
df_genres = df["Movie Genre"].str.get_dummies(sep = ",")
df = df.join(df_genres)
df = df.drop(columns=["Movie Genre"])

#oscar - nominee discrimination
df = Discriminate("Award",df)

#rotten status discrimination
df = Discriminate("Tomatometer Status",df)

#audience status discrimination
df = Discriminate("Audience Status",df) 

#content rating discrimination
df = Discriminate("Content Rating",df)

#nominee handling
nominees = df["Nominee"].tolist()

for index, i in enumerate(nominees):
    if i == 0:
        nominees[index] = 1

df["Nominee"] = nominees

#type casting
df["Audience Rating"] = df["Audience Rating"].astype(int)

#director labeling
df = EncodeLabel("Directors",df)

df.to_csv("new_oscars_data.csv", index = False)

oscar_winner_count = df["Winner"].sum()

print(oscar_winner_count)






