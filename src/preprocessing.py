# Generales
import pandas as pd
import numpy as np
# JSON y filtros de tiempo
import requests
from dateutil.relativedelta import relativedelta
# Escalado
from sklearn import preprocessing
import joblib

response = requests.get('https://www.datos.gov.co/resource/ceyp-9c7c.json')

trm_json_array = response.json()
trm_df = pd.DataFrame(trm_json_array)

trm_df.drop('vigenciadesde', axis=1, inplace=True)
trm_df.rename(columns = {'vigenciahasta':'vigencia'}, inplace = True)

trm_df['vigencia'] = trm_df['vigencia'].astype('datetime64[ns]')

latest_register = trm_df['vigencia'].max()
trm_df_3_years_ago = trm_df[(trm_df['vigencia'] > latest_register - relativedelta(years=3))]

trm_df_train = trm_df_3_years_ago[(trm_df_3_years_ago['vigencia'] < latest_register - relativedelta(months=6))]
trm_df_val = trm_df_3_years_ago[(trm_df_3_years_ago['vigencia'] >= latest_register - relativedelta(months=6)) & (trm_df_3_years_ago['vigencia'] < latest_register - relativedelta(months=3))]
trm_df_test = trm_df_3_years_ago[(trm_df_3_years_ago['vigencia'] >= latest_register - relativedelta(months=3))]

trm_df_train.drop('vigencia', axis=1, inplace=True);
trm_df_val.drop('vigencia', axis=1, inplace=True);
trm_df_test.drop('vigencia', axis=1, inplace=True);

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
trm_df_train_scaled = min_max_scaler.fit_transform(trm_df_train['valor'].values.reshape(-1, 1))
trm_df_val_scaled = min_max_scaler.transform(trm_df_val['valor'].values.reshape(-1, 1))
trm_df_test_scaled = min_max_scaler.transform(trm_df_test['valor'].values.reshape(-1, 1))
# Guardar el scaler ajustado
joblib.dump(min_max_scaler, 'scaler.pkl')

trm_df_train_scaled.tofile('trm_df_train_scaled.csv', sep = ',')
trm_df_val_scaled.tofile('trm_df_val_scaled.csv', sep = ',')
trm_df_test_scaled.tofile('trm_df_test_scaled.csv', sep = ',')