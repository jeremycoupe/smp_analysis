import psycopg2
import pandas.io.sql as psql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os.path

#conn = psycopg2.connect("dbname='fuserclt' user='fuserclt' password='fuserclt' host='localhost' ")

###########################
# conn = psycopg2.connect("dbname='fuser' user='fuser' password='fuser' host='localhost' ")

# cursor = conn.cursor()
# sql_file = open('smp_query.sql')

# df0 = psql.read_sql( sql_file.read() , conn)

# df0.to_csv('rophy_dmp_data.csv')
###########################

# df0 = pd.read_csv('/Users/wcoupe/Documents/git/smp_analysis/data/smp_data_example.csv',parse_dates=['start_time','end_time','eta_msg_time'])
# df0 = df0.rename(columns={"eta_msg_time": "record_timestamp"})

def plot_single_day_smp(df0):

	lead_time_parameter = 60

	lookahead_vec = []
	xtick_vec = []
	for i in range(6*lead_time_parameter + 1):
		if i % 6 == 0:
			lookahead_vec.append(6*lead_time_parameter - i)
			if (lookahead_vec[-1] / float(6)) % 5 == 0:
				xtick_vec.append(lookahead_vec[-1]/6)

	# print(xtick_vec)
	# print(lookahead_vec)



	unique_dmp = df0['dmp_id'].unique()

	for dmp_id in range(len(unique_dmp)):
		df = df0[df0['dmp_id']==unique_dmp[dmp_id]].reset_index()
		df.to_csv('/Users/wcoupe/Documents/git/smp_analysis/data/' + unique_dmp[dmp_id]+'.csv',index=False)

		if  df['dmp_status'].str.contains('ACTIVE').any():
			

			# start_accuracy = [[] for i in range(len(lookahead_vec))]
			# end_accuracy = [[] for i in range(len(lookahead_vec))]
			# count_accuracy = [[] for i in range(len(lookahead_vec))]
			# average_accuracy = [[] for i in range(len(lookahead_vec))]
			# max_accuracy = [[] for i in range(len(lookahead_vec))]



			plt.figure(dmp_id,figsize=(16,10))
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
			df = df.assign(xval= ( -reference_index + df.index + 6*lead_time_parameter) / 6 )
			for row in range(len(df['record_timestamp'])):
				df.loc[row,'lookahead'] = float(df.loc[row,'shift']) * -10
				df.loc[row,'lookahead2'] = pd.Timedelta( df.loc[row,'record_timestamp'] - reference_time ).total_seconds()
				if df.loc[row,'shift'] in lookahead_vec:
					df.loc[row,'start_prediction_accuracy'] = pd.Timedelta(df.loc[reference_index,'start_time'] - df.loc[row,'start_time']).total_seconds() / float(60)
					df.loc[row,'end_prediction_accuracy'] = pd.Timedelta( df.loc[len(df)-1,'end_time'] - df.loc[row,'end_time']	).total_seconds() / float(60)
					df.loc[row,'count_prediction_accuracy'] = df.loc[len(df)-1,'flight_count'] - df.loc[row,'flight_count']
					df.loc[row,'average_prediction_accuracy'] = ( df.loc[len(df)-1,'average_gate_hold'] - df.loc[row,'average_gate_hold'] ) / float(60*1000)
					df.loc[row,'max_prediction_accuracy'] = ( df.loc[len(df)-1,'max_gate_hold'] - df.loc[row,'max_gate_hold'] ) / float(60*1000)


			detected_smp = np.zeros(len(lookahead_vec))
			for lead_time in range(len(lookahead_vec)):
				df_temp = df[ df['shift'] == lookahead_vec[lead_time] ].reset_index()
				if len(df_temp) > 0:
					detected_smp[lead_time] = 1
			# 		start_accuracy[lead_time].append(pd.Timedelta( df.loc[reference_index,'start_time'] - df_temp.loc[0,'start_time']  ).total_seconds() / float(60) )
			# 		end_accuracy[lead_time].append(pd.Timedelta( df.loc[len(df)-1,'end_time'] - df_temp.loc[0,'end_time']  ).total_seconds() / float(60) )
			# 		count_accuracy[lead_time].append(  df.loc[len(df)-1,'flight_count'] - df_temp.loc[0,'flight_count']  )
			# 		average_accuracy[lead_time].append( ( df.loc[len(df)-1,'average_gate_hold'] - df_temp.loc[0,'average_gate_hold'] ) / float(60*1000) )
			# 		max_accuracy[lead_time].append( ( df.loc[len(df)-1,'max_gate_hold'] - df_temp.loc[0,'max_gate_hold'] ) / float(60*1000) )



			df.to_csv('/Users/wcoupe/Documents/git/smp_analysis/data/' + unique_dmp[dmp_id]+'_adjusted.csv',index=False)
			df = df[df['start_prediction_accuracy'].notnull()].reset_index()
			df.to_csv('/Users/wcoupe/Documents/git/smp_analysis/data/' + unique_dmp[dmp_id]+'_adjusted_predictions.csv',index=False)

			
			plt.subplot(3,2,1)
			plot_vec = detected_smp
			label = ' Detected SMP'
			xplot_vec = np.arange(len(detected_smp))
			plt.plot( xplot_vec , plot_vec)
			plt.title(df.loc[0,'resource_name'] + label)
			plt.xlim([-1,1+lead_time_parameter])
			plt.xticks( np.arange(lead_time_parameter+1,step=5),xtick_vec)
			ax = plt.gca()
			ax.yaxis.grid(True)



			plt.subplot(3,2,2)
			plot_vec = df['start_prediction_accuracy']
			label = ' <Actual Start - Predicted Start>'
			xplot_vec = df['xval']
			plt.plot( xplot_vec , plot_vec)
			plt.title(df.loc[0,'resource_name'] + label)
			plt.xlim([-1,1+lead_time_parameter])
			plt.xticks( np.arange(lead_time_parameter+1,step=5),xtick_vec)
			ax = plt.gca()
			ax.yaxis.grid(True)


			plt.subplot(3,2,3)
			plot_vec = df['end_prediction_accuracy']
			label = ' <Actual End - Predicted End>'
			xplot_vec = df['xval']
			plt.plot( xplot_vec , plot_vec)
			plt.title(df.loc[0,'resource_name'] + label)
			plt.xlim([-1,1+lead_time_parameter])
			plt.xticks( np.arange(lead_time_parameter+1,step=5),xtick_vec)
			ax = plt.gca()
			ax.yaxis.grid(True)


			plt.subplot(3,2,4)
			plot_vec = df['count_prediction_accuracy']
			label = ' <Actual Count Subject to SMP - Predicted Count Subject to SMP>'
			xplot_vec = df['xval']
			plt.plot( xplot_vec , plot_vec)
			plt.title(df.loc[0,'resource_name'] + label)
			plt.xlim([-1,1+lead_time_parameter])
			plt.xticks( np.arange(lead_time_parameter+1,step=5),xtick_vec)
			ax = plt.gca()
			ax.yaxis.grid(True)


			plt.subplot(3,2,5)
			plot_vec = df['average_prediction_accuracy']
			label = ' <Actual Mean Gate Hold - Predicted Mean Gate Hold>'
			xplot_vec = df['xval']
			plt.plot( xplot_vec , plot_vec)
			plt.title(df.loc[0,'resource_name'] + label)
			plt.xlim([-1,1+lead_time_parameter])
			plt.xticks( np.arange(lead_time_parameter+1,step=5),xtick_vec)
			ax = plt.gca()
			ax.yaxis.grid(True)

			plt.subplot(3,2,6)
			plot_vec = df['max_prediction_accuracy']
			label = ' <Actual Max Gate Hold - Predicted Max Gate Hold>'
			xplot_vec = df['xval']
			plt.plot( xplot_vec , plot_vec)
			plt.title(df.loc[0,'resource_name'] + label)
			plt.xlim([-1,1+lead_time_parameter])
			plt.xticks( np.arange(lead_time_parameter+1,step=5),xtick_vec)
			ax = plt.gca()
			ax.yaxis.grid(True)



			plt.tight_layout()
			plt.savefig('/Users/wcoupe/Documents/git/smp_analysis/figs/' + unique_dmp[dmp_id]+'.png')
			plt.close('all')



path = '/Users/wcoupe/Documents/Research/smp_predictions/code/debug2/'
allFiles = glob.glob(os.path.join(path, "**", "*.csv"),recursive=True)


for f in allFiles:
	print(f)
	df0 = pd.read_csv(f,parse_dates=['start_time','end_time','eta_msg_time'])
	df0 = df0.rename(columns={"eta_msg_time": "record_timestamp"})
	plot_single_day_smp(df0)