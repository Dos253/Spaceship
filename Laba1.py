import pandas as pd
from sklearn.preprocessing import StandardScaler

# Функция для обработки данных
def process_data(df):

    # 1. Заполнение пропущенных значений
    categorical_col = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']
    number_col = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    # Заполнение категориальных колонок модой
    for col in categorical_col:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Заполнение числовых колонок средним значением
    for col in number_col:
        df[col] = df[col].fillna(df[col].mean())

    # 2. Нормализация числовых данных
    scaler = StandardScaler()
    df[number_col] = scaler.fit_transform(df[number_col])

    # 3. Преобразование категориальных данных в численные (one-hot encoding)
    categorical_no_bool_col = list(filter(lambda x: x not in ['CryoSleep', 'VIP'], categorical_col))
    df = pd.get_dummies(df, columns=categorical_no_bool_col, prefix=categorical_no_bool_col)

    # 4. Преобразование всех булевых колонок в числовые (0 и 1)
    for col in df.columns:
        if df[col].dtype == bool:  # Если колонка булевая
            df[col] = df[col].astype(int)  # Преобразуем в 0 и 1

    # Если колонка 'Transported' существует (только в train.csv), преобразуем её в числовой формат
    if 'Transported' in df.columns:
        df['Transported'] = df['Transported'].astype(int)

    return df

# 1. Загрузка данных
train_path = 'train.csv'
test_path = 'test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Вывод информации о данных
print('DATA FROM TRAIN DATASET:')
print(train_df.head(), '\n')
print(train_df.info())

print('DATA FROM TEST DATASET:')
print(test_df.head(), '\n')
print(test_df.info())

# 2. Обработка train.csv
print("Processing train.csv...")
train_df = process_data(train_df)

# 3. Обработка test.csv
print("Processing test.csv...")
test_df = process_data(test_df)

# Вывод информации о пропущенных значениях после обработки
print('MISSING VALUES IN TRAIN DATASET AFTER FILLING:')
print(train_df.isnull().sum().sort_values(ascending=False), '\n')

print('MISSING VALUES IN TEST DATASET AFTER FILLING:')
print(test_df.isnull().sum().sort_values(ascending=False), '\n')

# 4. Сохранение обработанных данных
train_output_file = 'processed_train.csv'
test_output_file = 'processed_test.csv'

train_df.to_csv(train_output_file, index=False)
test_df.to_csv(test_output_file, index=False)

print(f'Processed train data saved to {train_output_file}')
print(f'Processed test data saved to {test_output_file}')