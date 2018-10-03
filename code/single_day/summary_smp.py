import psycopg2
import pandas.io.sql as psql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#conn = psycopg2.connect("dbname='fuserclt' user='fuserclt' password='fuserclt' host='localhost' ")
conn = psycopg2.connect("dbname='fuser' user='fuser' password='fuser' host='localhost' ")

cursor = conn.cursor()
sql_file = open('smp_query.sql')

df0 = psql.read_sql( sql_file.read() , conn)

df0.to_csv('rophy_dmp_data.csv')

unique_dmp = df0['dmp_id'].unique()

df_status = pd.DataFrame()
df_summary = pd.DataFrame()

idx=-1
for dmp_id in range(len(unique_dmp)):
	df = df0[df0['dmp_id']==unique_dmp[dmp_id]].reset_index()
	status = df['dmp_status'].unique()

	for dmp_status in range(len(status)):
		idx+=1
		df_status.loc[idx,'dmp_id'] = unique_dmp[dmp_id]
		df_status.loc[idx,'dmp_status'] = status[dmp_status]

	#df.to_csv('/Users/wcoupe/Documents/Research/smp_analysis/data/' + unique_dmp[dmp_id]+'.csv')

df_status.to_csv('/Users/wcoupe/Documents/Research/smp_analysis/data/dmp_status.csv')





unique_state = df_status['dmp_status'].unique()

idx=-1
for dmp_state in range(len(unique_state)):
	idx+=1
	df_summary.loc[idx,'dmp_state'] = unique_state[dmp_state]
	df_summary.loc[idx,'count'] = len( df_status[ df_status['dmp_status'] == unique_state[dmp_state] ] )


df_summary.to_csv('/Users/wcoupe/Documents/Research/smp_analysis/data/dmp_summary.csv',index=False)