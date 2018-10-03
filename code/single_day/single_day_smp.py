import psycopg2
import pandas.io.sql as psql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#conn = psycopg2.connect("dbname='fuserclt' user='fuserclt' password='fuserclt' host='localhost' ")

###########################
# conn = psycopg2.connect("dbname='fuser' user='fuser' password='fuser' host='localhost' ")

# cursor = conn.cursor()
# sql_file = open('smp_query.sql')

# df0 = psql.read_sql( sql_file.read() , conn)

# df0.to_csv('rophy_dmp_data.csv')
###########################

df0 = pd.read_csv('/Users/wcoupe/Documents/Research/smp_analysis/data/smp_data_example.csv',parse_dates=['start_time','end_time','eta_msg_time'])
df0 = df0.rename(columns={"eta_msg_time": "record_timestamp"})

lookahead_vec = []
xtick_vec = []
for i in range(181):
	if i % 6 == 0:
		lookahead_vec.append(180 - i)
		if (lookahead_vec[-1] / float(6)) % 5 == 0:
			xtick_vec.append(lookahead_vec[-1]/6)

print(xtick_vec)

unique_dmp = df0['dmp_id'].unique()

for dmp_id in range(len(unique_dmp)):
	df = df0[df0['dmp_id']==unique_dmp[dmp_id]].reset_index()
	df.to_csv('/Users/wcoupe/Documents/Research/smp_analysis/data/' + unique_dmp[dmp_id]+'.csv',index=False)

	if  df['dmp_status'].str.contains('ACTIVE').any():
		find_reference = True
		reference_time = -1
		reference_index = -1
		for row in range(len(df['record_timestamp'])):
			if df.loc[row,'dmp_status'] == 'ACTIVE':
				if find_reference:
					reference_time = df.loc[row,'start_time']
					reference_index = row
					find_reference = False


		df = df.assign(shift=reference_index - df.index )
		for row in range(len(df['record_timestamp'])):
			df.loc[row,'lookahead'] = float(df.loc[row,'shift']) * -10
			df.loc[row,'lookahead2'] = pd.Timedelta( df.loc[row,'record_timestamp'] - reference_time ).total_seconds()
			if df.loc[row,'shift'] in lookahead_vec:
				df.loc[row,'start_prediction_accuracy'] = df.loc[reference_index,'start_time'] - df.loc[row,'start_time']
				df.loc[row,'end_prediction_accuracy'] = df.loc[len(df)-1,'end_time'] - df.loc[row,'end_time']	
				df.loc[row,'count_prediction_accuracy'] = df.loc[reference_index,'flight_count'] - df.loc[row,'flight_count']
				df.loc[row,'average_prediction_accuracy'] = ( df.loc[len(df)-1,'average_gate_hold'] - df.loc[row,'average_gate_hold'] ) / float(60*1000)
				df.loc[row,'max_prediction_accuracy'] = ( df.loc[len(df)-1,'max_gate_hold'] - df.loc[row,'max_gate_hold'] ) / float(60*1000)


		df.to_csv('/Users/wcoupe/Documents/Research/smp_analysis/data/' + unique_dmp[dmp_id]+'_adjusted.csv',index=False)



