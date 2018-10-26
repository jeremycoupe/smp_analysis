import psycopg2
import pandas.io.sql as psql
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def f_smp_data(df0):
	unique_smp = df0['dmp_id'].unique()

	df_binary_status = pd.DataFrame()
	df_count_status = pd.DataFrame()
	df_summary = pd.DataFrame()

	status_vec = ['PROPOSED','AFFIRMED','REJECTED','ACTIVE','OBSOLETE','COMPLETED']

	idx=-1
	for dmp_id in range(len(unique_smp)):
		idx+=1
		df_temp = df0[df0['dmp_id']==unique_smp[dmp_id]].reset_index()
		df_temp.to_csv('/Users/wcoupe/Documents/git/smp_analysis/data/debug'+unique_smp[dmp_id]+'.csv')
		df_count_status.loc[idx,'dmp_id'] = unique_smp[dmp_id]
		df_binary_status.loc[idx,'dmp_id'] = unique_smp[dmp_id]
		for dmp_status in range(len(status_vec)):
			count_status = 0
			if df_temp.loc[0,'dmp_status'] == status_vec[dmp_status]:
				count_status +=1
			for row in range(1,len(df_temp)):
				if df_temp.loc[row,'dmp_status'] != df_temp.loc[row-1,'dmp_status']:
					if df_temp.loc[row,'dmp_status'] == status_vec[dmp_status]:
						count_status +=1
			df_count_status.loc[idx,status_vec[dmp_status]] = count_status

			if count_status > 0:
				df_binary_status.loc[idx,status_vec[dmp_status]] = True
			else:
				df_binary_status.loc[idx,status_vec[dmp_status]] = None

	
	for dmp_status in range(len(status_vec)):
		df_summary.loc[dmp_status,'dmp_status'] = status_vec[dmp_status]
		df_summary.loc[dmp_status,'count'] = len( df_binary_status[ df_binary_status[status_vec[dmp_status]] == True ] )


	df_count_status.to_csv('/Users/wcoupe/Documents/git/smp_analysis/data/smp_count_status_2018-10-24.csv',index=False)
	df_binary_status.to_csv('/Users/wcoupe/Documents/git/smp_analysis/data/smp_binary_status_2018-10-24.csv',index=False)
	df_summary.to_csv('/Users/wcoupe/Documents/git/smp_analysis/data/smp_summary_2018-10-24.csv',index=False)

def plot_single_day_smp(df0):

	lead_time_parameter = 60

	lookahead_vec = []
	xtick_vec = []
	for i in range(6*lead_time_parameter + 1):
		if i % 6 == 0:
			lookahead_vec.append(6*lead_time_parameter - i)
			if (lookahead_vec[-1] / float(6)) % 5 == 0:
				xtick_vec.append(lookahead_vec[-1]/6)

	unique_dmp = df0['dmp_id'].unique()

	for dmp_id in range(len(unique_dmp)):
		df = df0[df0['dmp_id']==unique_dmp[dmp_id]].reset_index()
		df.to_csv('/Users/wcoupe/Documents/git/smp_analysis/data/' + unique_dmp[dmp_id]+'.csv',index=False)

		if  df['dmp_status'].str.contains('ACTIVE').any():


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


			fake_flag = True
			df = df.assign(shift=reference_index - df.index )
			df = df.assign(xval= ( -reference_index + df.index + 6*lead_time_parameter) / 6 )
			for row in range(len(df['record_timestamp'])):
				df.loc[row,'lookahead'] = float(df.loc[row,'shift']) * -10
				df.loc[row,'lookahead2'] = pd.Timedelta( df.loc[row,'record_timestamp'] - reference_time ).total_seconds()
				df.loc[row,'xval2'] = lead_time_parameter + (df.loc[row,'lookahead2'] / float(60))
				
				if df.loc[row,'dmp_status'] == 'PROPOSED':
					df.loc[row,'status'] = 1
				if df.loc[row,'dmp_status'] == 'AFFIRMED':
					df.loc[row,'status'] = 2
				if df.loc[row,'dmp_status'] == 'OBSOLETE':
					df.loc[row,'status'] = 3
				if df.loc[row,'dmp_status'] == 'REJECTED':
					df.loc[row,'status'] = 4

				
				#if df.loc[row,'shift'] in lookahead_vec:
				if fake_flag:
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
			df = df[ (df['start_prediction_accuracy'].notnull())]
			#df = df.reset_index()
			df.to_csv('/Users/wcoupe/Documents/git/smp_analysis/data/' + unique_dmp[dmp_id]+'_adjusted_predictions_v2.csv',index=False)
			df = df[(df['xval2'] > 0)&(df['xval2'] < lead_time_parameter+1)].reset_index()
			

			plt.subplot(3,2,1)
			plot_vec = df['status']
			xplot_vec = df['xval2']
			label = ' Detected SMP'
			#xplot_vec = np.arange(len(detected_smp))
			plt.plot( xplot_vec , plot_vec)
			plt.title(df.loc[0,'resource_name'] + label)
			plt.xlim([-1,1+lead_time_parameter])
			plt.xticks( np.arange(lead_time_parameter+1,step=5),xtick_vec)
			plt.yticks([1,2,3,4],['PROPOSED','AFFIRMED','OBSOLETE','REJECTED'])
			ax = plt.gca()
			ax.yaxis.grid(True)



			plt.subplot(3,2,2)
			plot_vec = df['start_prediction_accuracy']
			label = ' <Actual Start - Predicted Start>'
			xplot_vec = df['xval2']
			plt.plot( xplot_vec , plot_vec)
			plt.title(df.loc[0,'resource_name'] + label)
			plt.xlim([-1,1+lead_time_parameter])
			plt.xticks( np.arange(lead_time_parameter+1,step=5),xtick_vec)
			ax = plt.gca()
			ax.yaxis.grid(True)


			plt.subplot(3,2,3)
			plot_vec = df['end_prediction_accuracy']
			label = ' <Actual End - Predicted End>'
			xplot_vec = df['xval2']
			plt.plot( xplot_vec , plot_vec)
			plt.title(df.loc[0,'resource_name'] + label)
			plt.xlim([-1,1+lead_time_parameter])
			plt.xticks( np.arange(lead_time_parameter+1,step=5),xtick_vec)
			ax = plt.gca()
			ax.yaxis.grid(True)


			plt.subplot(3,2,4)
			plot_vec = df['count_prediction_accuracy']
			label = ' <Actual Count Subject to SMP - Predicted Count Subject to SMP>'
			xplot_vec = df['xval2']
			plt.plot( xplot_vec , plot_vec)
			plt.title(df.loc[0,'resource_name'] + label)
			plt.xlim([-1,1+lead_time_parameter])
			plt.xticks( np.arange(lead_time_parameter+1,step=5),xtick_vec)
			ax = plt.gca()
			ax.yaxis.grid(True)


			plt.subplot(3,2,5)
			plot_vec = df['average_prediction_accuracy']
			label = ' <Actual Mean Gate Hold - Predicted Mean Gate Hold>'
			xplot_vec = df['xval2']
			plt.plot( xplot_vec , plot_vec)
			plt.title(df.loc[0,'resource_name'] + label)
			plt.xlim([-1,1+lead_time_parameter])
			plt.xticks( np.arange(lead_time_parameter+1,step=5),xtick_vec)
			ax = plt.gca()
			ax.yaxis.grid(True)

			plt.subplot(3,2,6)
			plot_vec = df['max_prediction_accuracy']
			label = ' <Actual Max Gate Hold - Predicted Max Gate Hold>'
			xplot_vec = df['xval2']
			plt.plot( xplot_vec , plot_vec)
			plt.title(df.loc[0,'resource_name'] + label)
			plt.xlim([-1,1+lead_time_parameter])
			plt.xticks( np.arange(lead_time_parameter+1,step=5),xtick_vec)
			ax = plt.gca()
			ax.yaxis.grid(True)



			plt.tight_layout()
			plt.savefig('/Users/wcoupe/Documents/git/smp_analysis/figs/' + unique_dmp[dmp_id]+'_lead_time_' + str(lead_time_parameter) + '.png')
			plt.close('all')

conn = psycopg2.connect("dbname='fuserclt' user='fuserclt' password='fuserclt' host='localhost' ")
cursor = conn.cursor()
sql_file = open('smp_query_2.sql')
df0 = psql.read_sql( sql_file.read() , conn)


f_smp_data(df0)
plot_single_day_smp(df0)

