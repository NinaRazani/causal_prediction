
from socket import close
from typing import Counter
from datatable import last
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import talib
import pandas as pd
from down_data import concat, extend_concat, fred_down, yfin_down, apply_talib_indicator
from encoder import create_transformer_encoder
from transfer_conc import transfer_ent

##download data 
#dataset
# major_fx = ['AUD=X', 'EUR=X', 'USDGBP=X', 'NZD=X', 'CAD=X', 'CHF=X', 'JPY=X']
# macto_fred_ids = ['M2SL', 'RBUSBIS', 'REAINTRATREARAT10Y', 'inflation']
# macro_yahoo_finance = [bac:BAC, gold:GC=F, wti:CL=F, brent:BZ=F, gas:NG=F] 
# indicators = [SMA, MACD, RSI] 

yah_df = yfin_down('USDCHF=X', '2004-01-01', '2023-12-31','1d', False) 
bac_df = yfin_down('BAC', '2004-01-01', '2023-12-31','1d', True) 
gold_df = yfin_down('GC=F', '2004-01-01', '2023-12-31','1d', True)
WTI_df = yfin_down('CL=F', '2004-01-01', '2023-12-31','1d', True)
brent_df = yfin_down('BZ=F', '2004-01-01', '2023-12-31','1d', True)
gas_df = yfin_down('NG=F', '2004-01-01', '2023-12-31','1d', True) 
sma_df = apply_talib_indicator(yah_df, talib.SMA, 'Close', 'SMA_20', timeperiod=20)   # type: ignore
macd_df = apply_talib_indicator(yah_df, talib.MACD, 'Close', ('MACD', 'MACD_signal', 'MACD_hist'),  # type: ignore
                                fastperiod=12, slowperiod=26, signalperiod=9) 
rsi_df = apply_talib_indicator(yah_df, talib.RSI, 'Close', 'RSI_14', timeperiod=14) #type: ignore 
def normalize_rsi(rsi):
    return (rsi - 50) / 50
rsi_df['RSI_14'] = rsi_df['RSI_14'].apply(normalize_rsi) 

real_df = fred_down('RBUSBIS', "date", "rate", '2004-01-01', '2023-12-31', True) 
MS_df = fred_down('M2SL', "date", "rate", '2004-01-01', '2023-12-31', True) 
interest_df = fred_down('REAINTRATREARAT10Y', "date", "rate", '2004-01-01', '2023-12-31', True) 
inflation_rate_df = fred_down('T10YIE', "date", "rate", '2004-01-01', '2023-12-31', True) 
# this lines done because the inflation is daily 
yah_df = yah_df.set_index('date')
inflation_rate_df = inflation_rate_df.set_index('date') 
result = yah_df.join(inflation_rate_df, how='inner')
yah_df = result[yah_df.columns].reset_index()
inflation_rate_df = result[inflation_rate_df.columns].reset_index()
# ##

# # collect transfer entropy
yah_bac = concat(bac_df, yah_df)
te_bac = transfer_ent(yah_bac, 365)
yah_gold = concat(gold_df, yah_df)
te_gold = transfer_ent(yah_gold, 365)
yah_wti = concat(WTI_df, yah_df)
te_wti = transfer_ent(yah_wti, 365)
yah_brent = concat(brent_df, yah_df)
te_brent = transfer_ent(yah_brent, 365)
yah_gas = concat(gas_df, yah_df) 
te_gas = transfer_ent(yah_gas, 365)  

merg_ms_yf_df = extend_concat(MS_df, yah_df, 'M')
te_ms = transfer_ent(merg_ms_yf_df, 365) 
merg_re_yf_df = extend_concat(real_df, yah_df, 'M')
te_re = transfer_ent(merg_re_yf_df, 365)
merg_ir_yf_df = extend_concat(interest_df, yah_df, 'M') 
te_ir = transfer_ent(merg_ir_yf_df, 365)
merg_inf_yf_df = extend_concat(inflation_rate_df, yah_df)
te_inf = transfer_ent(merg_inf_yf_df, 365) 

#add datetime
yah_df.reset_index(inplace=True)
yah_df['dayofweek'] = yah_df['date'].dt.day_of_week
yah_df['date'] = pd.to_datetime(yah_df['date'], format='%Y.%m')
yah_df['yearmonth'] = yah_df['date'].dt.month
yah_df['month'] = yah_df['date'].dt.month

# rename columns and reset index
merg_ms_yf_df.rename(columns={"rate": "rate_ms"}, inplace=True)
merg_re_yf_df.rename(columns={"rate":"rate_re"}, inplace=True)
merg_ir_yf_df.rename(columns={"rate":"rate_ir"}, inplace=True) 
merg_inf_yf_df.rename(columns={"rate":"rate_inf"}, inplace=True) 

te_ms.rename(columns={"mean_flow":"ms_flow"}, inplace=True)
te_re.rename(columns={"mean_flow":"re_flow"}, inplace=True)
te_ir.rename(columns={"mean_flow":"ir_flow"}, inplace=True)
te_inf.rename(columns={"mean_flow":"inf_flow"}, inplace=True)
te_bac.rename(columns={"mean_flow":"bac_flow"}, inplace=True)
te_gold.rename(columns={"mean_flow":"gold_flow"}, inplace=True)
te_wti.rename(columns={"mean_flow":"wti_flow"}, inplace=True)
te_brent.rename(columns={"mean_flow":"brent_flow"}, inplace=True)
te_gas.rename(columns={"mean_flow":"gas_flow"}, inplace=True)

yah_bac.rename(columns={col.strip(): "bac" for col in yah_wti.columns if col.strip() == "Close_x"}, inplace=True)
yah_brent.rename(columns={col.strip(): "brent" for col in yah_wti.columns if col.strip() == "Close_x"}, inplace=True)
yah_gas.rename(columns={col.strip(): "gas" for col in yah_wti.columns if col.strip() == "Close_x"}, inplace=True)
yah_gold.rename(columns={col.strip(): "gold" for col in yah_wti.columns if col.strip() == "Close_x"}, inplace=True)
yah_wti.rename(columns={col.strip(): "wti" for col in yah_wti.columns if col.strip() == "Close_x"}, inplace=True)

##

te_ms = te_ms.reset_index(drop=True)
te_re = te_re.reset_index(drop=True)
te_ir= te_ir.reset_index(drop=True)
te_inf = te_inf.reset_index(drop=True) 

te_bac = te_bac.reset_index(drop=True)
te_gold = te_gold.reset_index(drop=True)
te_wti = te_wti.reset_index(drop=True)
te_brent = te_brent.reset_index(drop=True) 
te_gas = te_gas.reset_index(drop=True)

# #concatenate dataframes 
row_to_remove = merg_ms_yf_df.shape[0] - te_ms.shape[0]

last_df = pd.concat([te_ms['ms_flow'], te_re['re_flow'],te_ir['ir_flow'], te_inf['inf_flow'],
                     te_bac['bac_flow'], te_gold['gold_flow'], te_wti['wti_flow'], te_brent['brent_flow'], te_gas['gas_flow']], axis=1)

last_df.reset_index(inplace=True)
last_df.drop(['index'], axis=1, inplace=True) 

dfs_to_add = [merg_re_yf_df, merg_ir_yf_df, merg_inf_yf_df, merg_ms_yf_df, yah_bac, yah_gold, yah_wti, yah_brent, yah_gas]
dfs_to_add2 = [sma_df, macd_df, rsi_df]

for i, df in enumerate(dfs_to_add, 1): 

    df = df.iloc[row_to_remove:, :]  
    column_name = df.columns[0]  # Get the name of the first column
    last_df[column_name] = df[column_name].values 


for i, df in enumerate(dfs_to_add2, 1): 

    df = df.iloc[row_to_remove:, :]  
    column_name = df.columns[2]  # Get the name of the first column
    last_df[column_name] = df[column_name].values 
m = []
s = []
for i in range(len(yah_df)): 
    close_column = yah_df.iloc[:i, 1].values
    S = np.std(close_column)  # type: ignore 
    s.append(S)
    M = np.mean(close_column) # type: ignore   
    m.append(M) 

yah_df['Std'] = s
yah_df['Mean'] = m

yah_df = yah_df.iloc[row_to_remove:, :] 
yah_df.reset_index(inplace=True)

last_df['dayweek'] = yah_df['dayofweek']
last_df['monthyear'] = yah_df['yearmonth']
last_df['Mean'] = yah_df['Mean']
last_df['Std'] = yah_df['Std']
last_df['target'] = yah_df['Close'] 

print(last_df.head(73))

# # #analysis the data
# def array_stats(arr): 
    
#     # Calculate statistics
#     stats = {
#         "Mean": np.mean(arr),
#         "Median": np.median(arr),
#         "Standard Deviation": np.std(arr),
#         "Minimum": np.min(arr),
#         "Maximum": np.max(arr),
#         "25th Percentile": np.percentile(arr, 25),
#         "50th Percentile": np.percentile(arr, 50),
#         "75th Percentile": np.percentile(arr, 75),
#         "Interquartile Range": np.percentile(arr, 75) - np.percentile(arr, 25),
#         "Skewness": float(arr.skew()) if hasattr(arr, 'skew') else None,
#         "Kurtosis": float(arr.kurtosis()) if hasattr(arr, 'kurtosis') else None
#     }
    
#     return stats
# stats = array_stats(last_column_array) 
# # Print the statistics
# for stat, value in stats.items():
#     if value is not None:
#         print(f"{stat}: {value:.4f}") 
#     else:
#         print(f"{stat}: Not available") 

# ####

# #####prediction###########

# # # transformer encoder 

# # # Number of previous days to use for prediction
window_size = 72 #last three month 
# Create the feature matrix with the previous 3 month
X = []
y = []

for i in range(window_size, window_size+1): 
    print(i)
    X.append(last_df.iloc[i-window_size:i, :-3].values.flatten()) 
    # # y.append(1 if last_df.iloc[i, -1] > last_df.iloc[i-1, -1] else 0)  # type: ignore # 1 if price went up, 0 if down  
    # # y.append(0 if last_df.iloc[i, -1] < Mean-Std else 1 if last_df.iloc[i, -1] > Mean+Std else 2)      # 0 if short, 1 if long, 2 neutral
    y.append(1 if (last_df.iloc[i, -1] < last_df.iloc[i, -3] - last_df.iloc[i, -2] or last_df.iloc[i, -1] >
                    last_df.iloc[i, -3] + last_df.iloc[i, -2]) else 0)  # type: ignore 1 if close>mean(t-1)-std(t-1)
    print(last_df.iloc[i, -1], last_df.iloc[i, -2], last_df.iloc[i, -3])

X = np.array(X, dtype=np.float32) 
y = np.array(y, dtype=np.int32) 
print(X[0], y[0]) 

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# # # smote = SMOTE(random_state=42)
# # # X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train) # type: ignore   
# # # X_test_smote, y_test_smote = smote.fit_resample(X_test, y_test) # type: ignore

# # # Create and compile the transformer encoder

# input_shape = X.shape[1]  
# encoder = create_transformer_encoder(input_shape)
# encoder.compile(optimizer='adam', loss='mse')  

# # Use the encoder to get the new feature representations
# X_encoded = encoder.predict(X) 

# # Create a pipeline with StandardScaler and SVM
# svm_pipeline = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))

# # Train the SVM classifier
# # svm_pipeline.fit(X_train_smote, y_train_smote)
# svm_pipeline.fit(X_train, y_train)

# # Make predictions
# # y_pred = svm_pipeline.predict(X_test_smote)
# y_pred = svm_pipeline.predict(X_test)

# # Evaluate the model
# # accuracy = accuracy_score(y_test_smote, y_pred)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))    


# random forest classification
#just predict using now features (not previous features) 
# file_path = "data.txt"
# def append_to_file(value):
#     with open(file_path, "a") as file:
#         file.write(str(value) + "\n")

# X = last_df.iloc[:, :23] 
# y = []
# for i in range(0, len(last_df)):
#     y.append(0 if last_df.iloc[i, -1] <= Mean+Std else 1 if last_df.iloc[i, -1] > Mean+Std else 2)  

# X = np.array(X)
# y = np.array(y) 
# feature_name = ['ms_flow', 're_flow', 'ir_flow', 'inf_flow', 'bac_flow', 'gold_flow',
#                    'wti_flow', 'brent_flow', 'gas_flow', 'rate_re', 'rate_ir', 'rate_inf',
#                    'rate_ms', 'bac', 'gold', 'wti', 'brent', 'gas', 'SMA_20', 'MACD',
#                    'RSI_14', 'dayweek', 'monthyear']   
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 


# rf_model = RandomForestClassifier(n_estimators=1000, random_state=42) 
# rf_model.fit(X_train, y_train)

# # Calculate accuracy
# y_pred = rf_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy: {accuracy:.4f}" )

# #calculate the importance of features 
# rf_importance = pd.DataFrame({
#     'feature': feature_name,
#     'importance': rf_model.feature_importances_
# }).sort_values('importance', ascending=False)

# print(rf_importance) 
# #get the result of cross validation
# # cv_scores = cross_val_score(rf_model, X_train, y_train, cv=10) 
# # print(f"Cross-validation scores: {cv_scores}")
# # print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")  
# # append_to_file(accuracy)
# # append_to_file(rf_importance) 
# # append_to_file(cv_scores)
# # append_to_file(cv_scores.mean())
# # append_to_file(cv_scores.std()*2)  

# # write the strategy for volatility output(on the prediction y_pred)

# def get_matching_fraction(df, column_name, match_array):
 
#     n_rows = len(match_array)
    
#     # Ensure we're not asking for more rows than exist in the DataFrame
#     if n_rows > len(df):
#         raise ValueError("match_array is larger than the DataFrame")
    
#     # Return the last n_rows of the specified column
#     return df[column_name].tail(n_rows)


# y_close = get_matching_fraction(yah_df, 'Close', y_pred)  

# def volatility_strategy(close_prices, volatility_predictions, risk_free_rate=0.02):
#     # Calculate daily returns
#     daily_returns = np.diff(close_prices) / close_prices[:-1]
    
#     # Generate trading signals (1 for long, 0 for no position)
#     signals = volatility_predictions[:-1]  # Align with returns
    
#     # Calculate strategy returns
#     strategy_returns = daily_returns * signals
    
#     # Calculate performance metrics
#     total_return = np.prod(1 + strategy_returns) - 1
#     avg_daily_return = np.mean(strategy_returns)
#     std_daily_return = np.std(strategy_returns)
    
#     # Calculate Sharpe ratio (annualized)
#     sharpe_ratio = (avg_daily_return - risk_free_rate/252) / std_daily_return * np.sqrt(252)
    
#     return sharpe_ratio, total_return, signals

# print(volatility_strategy(y_close, y_pred))