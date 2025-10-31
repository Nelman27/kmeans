import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logs = [
  {"timestamp": "2025-10-27T10:00:00Z", "level": "INFO",  "service": "api-usuarios", "msg": "User login successful",                         "status": 200},
  {"timestamp": "2025-10-27T10:01:15Z", "level": "ERROR", "service": "api-pagos",    "msg": "Failed to connect to primary database",          "status": 503},
  {"timestamp": "2025-10-27T10:03:30Z", "level": "WARN",  "service": "api-externa",  "msg": "Request timeout after 30s for provider 'X'",     "status": 504}
]

# 1) DataFrame
df = pd.DataFrame(logs)
print('DataFrame inicial:')
print(df, '\n')

# 2) Normalizar columnas
df['status'] = df['status'].fillna(200).astype(int)
df['level'] = df['level'].fillna('unknown')
df['service'] = df['service'].fillna('unknown')

# 3) Flags desde msg
df['has_timeout'] = df['msg'].apply(lambda x: 1 if isinstance(x, str) and 'timeout' in x.lower() else 0)
df['has_db_error'] = df['msg'].apply(lambda x: 1 if isinstance(x, str) and 'db' in x.lower() else 0)

print('Después de crear flags:')
print(df[['status','level','service','has_timeout','has_db_error']], '\n')

# 4) One-Hot
categorical_cols = ['level','service']
enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
enc_arr = enc.fit_transform(df[categorical_cols])
enc_df = pd.DataFrame(enc_arr, columns=enc.get_feature_names_out(categorical_cols))

print('One-Hot columns:')
print(enc_df.head(), '\n')

# 5) Concatenar y escalar
numeric_df = df[['status','has_timeout','has_db_error']].reset_index(drop=True)
enc_df = enc_df.reset_index(drop=True)
final = pd.concat([numeric_df, enc_df], axis=1)
print('Final features (antes de escalar):')
print(final, '\n')

scaler = StandardScaler()
features_scaled = scaler.fit_transform(final)
print('Final features escaladas (primeras filas):')
print(features_scaled[:3])