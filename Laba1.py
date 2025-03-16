import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

df = pd.read_csv('train.csv')
print('ДАННЫЕ ИЗ НАБОРА ДАННЫХ:')
print(df.head(), '\n')
print(df.info())

df[["Deck", "Cabin_num", "Side"]] = df["Cabin"].str.split("/", expand=True)
df['Cabin_num'] = df['Cabin_num'].astype(float)
try:
    df = df.drop('Cabin', axis=1)
    df = df.drop('Name', axis=1)
except KeyError:
    print("Field does not exist")

missing_values = df.isnull().sum().sort_values(ascending=False)
print('ОТСУТСТВУЮЩИЕ ЗНАЧЕНИЯ ПЕРЕД ЗАПОЛНЕНИЕМ:')
print(missing_values, '\n')

print('DATA FROM DATASET AFTER TRANSFORMATION:')
print(df.head(), '\n')

categorical_col = ['HomePlanet', 'CryoSleep', 'Deck', 'Side', 'Destination', 'VIP']
number_col = ['Cabin_num', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

df['Age'] = df['Age'].fillna(df['Age'].median())
df['RoomService'] = df['RoomService'].fillna(df['RoomService'].std())

for col in categorical_col:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in number_col:
    df[col] = df[col].fillna(df[col].mean())

missing_values = df.isnull().sum().sort_values(ascending=False)
print('MISSING VALUES AFTER FILLING:')
print(missing_values, '\n')

categorical_no_bool_col=list(filter(lambda x:x!='CryoSleep'and x!='VIP',categorical_col))

df['CryoSleep'] = df['CryoSleep'].astype(int)
df['VIP'] = df['VIP'].astype(int)
df['Transported'] = df['Transported'].astype(int)

scaler = StandardScaler()
df[number_col] = scaler.fit_transform(df[number_col])
df['Age'] = scaler.fit_transform(df[['Age']])
df['RoomService'] = scaler.fit_transform(df[['RoomService']])
df = pd.get_dummies(df, columns=categorical_no_bool_col, prefix=categorical_no_bool_col)
df = df.replace({True: 1, False: 0})

df.to_csv("processed_train.csv", index=False)