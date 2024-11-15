



import pandas as pd
import numpy as np

df = pd.read_parquet("times_data.parquet")
df.drop(columns=["line", "PrimaryRoadName", "IntervalStart"], inplace=True)

def fill_na_with_mode(group):
    mode_value = group.mode().iloc[0]  # Get the most frequent value in the group
    return group.fillna(mode_value)  # Fill NA values with the mode

df['StartEndDescription'] = df.groupby('Id')['StartEndDescription'].transform(fill_na_with_mode)

df.fillna(method="ffill")

def categorical_to_onehot(df):
    # Create a copy of the original dataframe
    df_encoded = df.copy()

    for column in ["DataStatus", "Name", "StartEndDescription", "Timezone"]:
        # Get unique values and sort them
        unique_values = sorted(df[column].unique())

        # Create a dictionary to map categories to numbers
        category_dict = {value: i for i, value in enumerate(unique_values)}

        # Convert categories to numbers
        df_encoded[column] = df[column].map(category_dict)

        # Create one-hot encoded columns
        for value in unique_values:
            new_column = f"{column}_{category_dict[value]}"
            df_encoded[new_column] = (df[column] == value).astype(int)

        # Remove the original column
        df_encoded = df_encoded.drop(column, axis=1)

    return df_encoded

df = categorical_to_onehot(df)

from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(df, test_size=0.1, shuffle=False)

train_data.to_parquet("processed_train_data.parquet")
test_data.to_parquet("processed_test_data.parquet")

"""# Load Data

### Load data for table format
"""

import pandas as pd
import numpy as np
import sklearn
import torch

from sklearn.model_selection import train_test_split

train_data, test_data = pd.read_parquet("processed_train_data.parquet"), pd.read_parquet("processed_test_data.parquet")

x_train, y_train = train_data.drop(columns=["ExcessDelay"]), train_data[["ExcessDelay"]]
x_test, y_test = test_data.drop(columns=["ExcessDelay"]), test_data[["ExcessDelay"]]

"""### Load data for time-series format"""

import pandas as pd
import numpy as np
import sklearn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

train_data, test_data = pd.read_parquet("processed_train_data.parquet"), pd.read_parquet("processed_test_data.parquet")

def process_dataframe(df, M, N):
    # Ensure time columns are in the correct data type
    time_cols = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']
    for col in time_cols:
        df[col] = pd.to_numeric(df[col])

    # Group by 'Id' and sort by time
    grouped = df.groupby('Id')
    sorted_groups = [group.sort_values(time_cols) for _, group in grouped]

    # Function to create windows
    def create_windows(group):
        windows = []
        for i in range(0, len(group) - M + 1, N):
            windows.append(group.iloc[i:i+M])
        return windows

    # Create windows for each group
    all_windows = [window for group in sorted_groups for window in create_windows(group)]

    # Join all windows
    dataset = pd.concat(all_windows, ignore_index=True)

    # Split into X and y
    y = dataset['ExcessDelay'].values
    X = dataset.values #.drop('ExcessDelay', axis=1).values

    # Reshape X and y
    num_samples = len(X) // M
    num_features = X.shape[1]
    X = X.reshape(num_samples, M, num_features)
    y = y.reshape(num_samples, M)

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    return X_tensor[:, :-1, :], y_tensor[:, 1:]

# Usage example:
M = 300
N = 100
x_train, y_train = process_dataframe(train_data, M=M, N=N)
x_test, y_test = process_dataframe(test_data, M=M, N=N)

train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
val_dataset = torch.utils.data.TensorDataset(x_test, y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

"""# Table algorithm

## XGBoost
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

model = xgb.XGBRegressor(random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared Score: {r2}")

# Feature importance
feature_importance = model.feature_importances_
feature_importance_dict = dict(zip(train_data.columns, feature_importance))
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# top 10 most important feature
print("\nFeature Importance:")
for feature, importance in sorted_features[:10]:
    print(f"{feature}: {importance}")

# save model
model.save_model("xgboost_excess_delay_model.model")

# load model
loaded_model = xgb.XGBRegressor()
loaded_model.load_model("xgboost_excess_delay_model.model")

"""## LightGBM"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb

model = lgb.LGBMRegressor(random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("LightGBM Results:")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared Score: {r2}")

# Feature importance
feature_importance = model.feature_importances_
feature_importance_dict = dict(zip(train_data.columns, feature_importance))
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

print("\nFeature Importance:")
for feature, importance in sorted_features[:10]:  # Print top 10 features
    print(f"{feature}: {importance}")

# save model
model.booster_.save_model("lightgbm_excess_delay_model.txt")

# To load the model later, you can use:
loaded_model = lgb.Booster(model_file='lightgbm_excess_delay_model.txt')

"""## SVM"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import joblib

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

model = SVR(kernel='rbf')  # You can experiment with different kernels
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)



"""## Catboost"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool

cat_features = train_data.keys().to_list()

cat_features = [feature for feature in cat_features if feature != 'ExcessDelay']

train_pool = Pool(x_train, y_train, cat_features=cat_features)
test_pool = Pool(x_test, y_test, cat_features=cat_features)

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=15,
    loss_function='RMSE',
    verbose=20,  # Print training progress every 100 iterations,
    task_type="GPU",
    devices='0',
)
model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=100)

y_pred = model.predict(test_pool)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("CatBoost Results:")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared Score: {r2}")

# Feature importance
feature_importance = model.feature_importances_
feature_names = train_data.columns
feature_importance_dict = dict(zip(feature_names, feature_importance))
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

print("\nFeature Importance:")
for feature, importance in sorted_features[:10]:  # Print top 10 features
    print(f"{feature}: {importance}")

# Save the model for later use
model.save_model("catboost_excess_delay_model.cbm")

# To load the model later, you can use:
loaded_model = CatBoostRegressor()
loaded_model.load_model("catboost_excess_delay_model.cbm")

"""# Time series algorithm

## LSTM
"""

import torch
from torch import nn

class LSTMRegression(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMRegression, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Training function
def train_lstm(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

input_size = x_train.shape[2]  # Number of features
hidden_size = 512
num_layers = 10
output_size = 1

model = LSTMRegression(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
num_epochs = 50

train_lstm(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

print("Training completed!")

"""## GRU"""

import torch
from torch import nn

class GRURegression(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRURegression, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train_gru(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
        model.to(device)

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

input_size = x_train.shape[2]  # Number of features
hidden_size = 128
num_layers = 5
output_size = 1

model = GRURegression(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 50

train_gru(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

print("Training completed!")

