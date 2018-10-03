import psycopg2
import pandas.io.sql as psql
import numpy as np
import pandas as pd
from pandas.tools.plotting import table
import matplotlib.pyplot as plt
import math

exempt_window_size = 60

db_string = str(exempt_window_size) + '_minute_exempt_hacked_eta'
database_name = "dbname= '%s' user='fuser' password='fuser' host='localhost'  " %db_string

conn = psycopg2.connect(database_name)

runway_vector = ['18C','18L','36C','36R']

plt.figure(10,figsize=(12,10))
plt.figure(11,figsize=(12,10))

lookahead_vec = []
xtick_vec = []
for i in range(181):
	if i % 6 == 0:
		lookahead_vec.append(180 - i)
		if (lookahead_vec[-1] / float(6)) % 5 == 0:
			xtick_vec.append(lookahead_vec[-1]/6)

print(xtick_vec)

for rwy in range(len(runway_vector)):
	q = '''SELECT
	*
	FROM 
	scheduler_analysis_dmp
	where
	dmp_status = 'ACTIVE'
	and resource_name = '%s'
	'''%runway_vector[rwy]
	df0 = psql.read_sql(q, conn)
	dmp_vec = np.unique(df0['dmp_id'])

	print(len(dmp_vec))

	start_accuracy = [[] for i in range(len(lookahead_vec))]
	end_accuracy = [[] for i in range(len(lookahead_vec))]
	count_accuracy = [[] for i in range(len(lookahead_vec))]
	average_accuracy = [[] for i in range(len(lookahead_vec))]
	max_accuracy = [[] for i in range(len(lookahead_vec))]
	
	advance_notice = []
	actual_notice = []
	for dmp in range(len(dmp_vec)):
	#for dmp in range(2):
		print(dmp_vec[dmp])
		q = '''SELECT
		*
		FROM 
		scheduler_analysis_dmp
		where
		dmp_id = '%s'
		Order by msg_time ASC
		'''%dmp_vec[dmp]

		df = psql.read_sql(q, conn)

		find_reference = True
		reference_time = -1
		reference_index = -1
		for row in range(len(df['msg_time'])):
			if df.loc[row,'dmp_status'] == 'ACTIVE':
				if find_reference:
					reference_time = df.loc[row,'start_time']
					reference_index = row
					find_reference = False


		advance_notice.append( pd.Timedelta( df.loc[0,'start_time'] - df.loc[0,'creation_time'] ).total_seconds() / float(60) )

		

		df = df.assign(shift=reference_index - df.index )
		for row in range(len(df['msg_time'])):
			df.loc[row,'lookahead'] = float(df.loc[row,'shift']) * -10
			if df.loc[row,'shift'] in lookahead_vec:
				df.loc[row,'start_prediction_accuracy'] = df.loc[reference_index,'start_time'] - df.loc[row,'start_time']
				df.loc[row,'end_prediction_accuracy'] = df.loc[len(df)-1,'end_time'] - df.loc[row,'end_time']	
				df.loc[row,'count_prediction_accuracy'] = df.loc[reference_index,'flight_count'] - df.loc[row,'flight_count']
				df.loc[row,'average_prediction_accuracy'] = ( df.loc[len(df)-1,'average_gate_hold'] - df.loc[row,'average_gate_hold'] ) / float(60*1000)
				df.loc[row,'max_prediction_accuracy'] = ( df.loc[len(df)-1,'max_gate_hold'] - df.loc[row,'max_gate_hold'] ) / float(60*1000)

		df.to_csv('debug/' + str(dmp_vec[dmp])+'.csv')

		actual_notice.append( pd.Timedelta( df.loc[reference_index,'start_time'] - df.loc[0,'creation_time'] ).total_seconds() / float(60) )

		for lead_time in range(len(lookahead_vec)):
			df_temp = df[ df['shift'] == lookahead_vec[lead_time] ].reset_index()
			if len(df_temp) > 0:
				start_accuracy[lead_time].append(pd.Timedelta( df.loc[reference_index,'start_time'] - df_temp.loc[0,'start_time']  ).total_seconds() / float(60) )
				end_accuracy[lead_time].append(pd.Timedelta( df.loc[len(df)-1,'end_time'] - df_temp.loc[0,'end_time']  ).total_seconds() / float(60) )
				count_accuracy[lead_time].append(  df.loc[reference_index,'flight_count'] - df_temp.loc[0,'flight_count']  )
				average_accuracy[lead_time].append( ( df.loc[len(df)-1,'average_gate_hold'] - df_temp.loc[0,'average_gate_hold'] ) / float(60*1000) )
				max_accuracy[lead_time].append( ( df.loc[len(df)-1,'max_gate_hold'] - df_temp.loc[0,'max_gate_hold'] ) / float(60*1000) )

	#print(start_accuracy)
	#print(max_accuracy)

	count_of_predictions = np.zeros(len(lookahead_vec))
	mean_start_accuracy = np.zeros(len(lookahead_vec))
	std_start_accuracy = np.zeros(len(lookahead_vec))

	mean_end_accuracy = np.zeros(len(lookahead_vec))
	std_end_accuracy = np.zeros(len(lookahead_vec))

	mean_count_accuracy = np.zeros(len(lookahead_vec))
	std_count_accuracy = np.zeros(len(lookahead_vec))

	mean_average_accuracy = np.zeros(len(lookahead_vec))
	std_average_accuracy = np.zeros(len(lookahead_vec))

	mean_max_accuracy = np.zeros(len(lookahead_vec))
	std_max_accuracy = np.zeros(len(lookahead_vec))



	for lead_time in range(len(lookahead_vec)):
		count_of_predictions[lead_time] = len(start_accuracy[lead_time])
		
		### calculate mean and std for start 
		mean_start_accuracy[lead_time] = np.mean(start_accuracy[lead_time])
		std_start_accuracy[lead_time] = np.std(start_accuracy[lead_time])

		### calculate mean and std for end
		mean_end_accuracy[lead_time] = np.mean(end_accuracy[lead_time])
		std_end_accuracy[lead_time] = np.std(end_accuracy[lead_time])

		### calculate mean and std for count
		mean_count_accuracy[lead_time] = np.mean(count_accuracy[lead_time])
		std_count_accuracy[lead_time] = np.std(count_accuracy[lead_time])

		### calculate mean and std for average
		mean_average_accuracy[lead_time] = np.mean(average_accuracy[lead_time])
		std_average_accuracy[lead_time] = np.std(average_accuracy[lead_time])

		### calculate mean and std for max
		mean_max_accuracy[lead_time] = np.mean(max_accuracy[lead_time])
		std_max_accuracy[lead_time] = np.std(max_accuracy[lead_time])


	plt.figure(figsize=(16,10))
		
	xvec = np.arange(len(mean_start_accuracy))
	xvec2 = np.arange(len(mean_start_accuracy),step=5)


	plt.subplot(3,2,1)
	
	plt.plot(xvec,count_of_predictions)
	plt.title(runway_vector[rwy] + ' Count of Predictions - ' + str(exempt_window_size) + ' Exempt Window Size')
	plt.xlim([-1,31])
	plt.xticks(xvec2,xtick_vec  )
	ax = plt.gca()
	ax.yaxis.grid(True)




	plt.subplot(3,2,2)

	for lead_time in range(len(lookahead_vec)):
		for v in range(len(start_accuracy[lead_time])):
			plt.plot(lead_time,start_accuracy[lead_time][v],marker='o',color = 'blue',alpha=0.11)


	plt.plot(xvec,mean_start_accuracy,linewidth = 3,color='blue')
	plt.fill_between(xvec,mean_start_accuracy+std_start_accuracy,mean_start_accuracy - std_start_accuracy, alpha=0.4	 )
	plt.title(runway_vector[rwy] + ' <Actual Start - Predicted Start>')
	plt.xlim([-1,31])
	plt.xticks(xvec2,xtick_vec  )
	ax = plt.gca()
	ax.yaxis.grid(True)




	plt.subplot(3,2,3)

	for lead_time in range(len(lookahead_vec)):
		for v in range(len(end_accuracy[lead_time])):
			plt.plot(lead_time,end_accuracy[lead_time][v],marker='o',color = 'blue',alpha=0.11)



	yvec_mean = mean_end_accuracy
	yvec_std = std_end_accuracy
	
	plt.plot(xvec,yvec_mean,linewidth = 3,color='blue')
	plt.fill_between(xvec,yvec_mean + yvec_std,yvec_mean - yvec_std, alpha=0.4	 )
	plt.title(runway_vector[rwy] + ' <Actual End - Predicted End>')
	plt.xlim([-1,31])
	plt.xticks(xvec2,xtick_vec  )
	ax = plt.gca()
	ax.yaxis.grid(True)

	plt.subplot(3,2,4)

	for lead_time in range(len(lookahead_vec)):
		for v in range(len(count_accuracy[lead_time])):
			plt.plot(lead_time,count_accuracy[lead_time][v],marker='o',color = 'blue',alpha=0.11)


	yvec_mean = mean_count_accuracy
	yvec_std = std_count_accuracy
	
	plt.plot(xvec,yvec_mean,linewidth = 3,color='blue')
	plt.fill_between(xvec,yvec_mean + yvec_std,yvec_mean - yvec_std, alpha=0.4	 )
	plt.title(runway_vector[rwy] + ' <Actual Count Subject to SMP - Predicted Count Subject to SMP>')
	plt.xlim([-1,31])
	plt.xticks(xvec2,xtick_vec  )
	ax = plt.gca()
	ax.yaxis.grid(True)

	plt.subplot(3,2,5)

	for lead_time in range(len(lookahead_vec)):
		for v in range(len(average_accuracy[lead_time])):
			plt.plot(lead_time,average_accuracy[lead_time][v],marker='o',color = 'blue',alpha=0.11)


	yvec_mean = mean_average_accuracy
	yvec_std = std_average_accuracy
	
	plt.plot(xvec,yvec_mean,linewidth = 3,color='blue')
	plt.fill_between(xvec,yvec_mean + yvec_std,yvec_mean - yvec_std, alpha=0.4	 )
	plt.title(runway_vector[rwy] + ' <Actual Mean Gate Hold - Predicted Mean Gate Hold>')
	plt.xlim([-1,31])
	plt.xticks(xvec2,xtick_vec  )
	ax = plt.gca()
	ax.yaxis.grid(True)


	plt.subplot(3,2,6)

	for lead_time in range(len(lookahead_vec)):
		for v in range(len(max_accuracy[lead_time])):
			plt.plot(lead_time,max_accuracy[lead_time][v],marker='o',color = 'blue',alpha=0.11)


	yvec_mean = mean_max_accuracy
	yvec_std = std_max_accuracy
	
	plt.plot(xvec,yvec_mean,linewidth = 3,color='blue')
	plt.fill_between(xvec,yvec_mean + yvec_std,yvec_mean - yvec_std, alpha=0.4	 )
	plt.title(runway_vector[rwy] + ' <Actual Max Gate Hold - Predicted Max Gate Hold>')
	plt.xlim([-1,31])
	plt.xticks(xvec2,xtick_vec  )
	ax = plt.gca()
	ax.yaxis.grid(True)

	



	plt.tight_layout()
	plt.savefig('figs/'+runway_vector[rwy]+'_smp_prediction_accuracy_extension_planning_' + str(exempt_window_size) + '_minute_exempt.png')

	plt.figure(10)
	plt.subplot(2,2,rwy+1)
	plt.hist(advance_notice,range=[0,30],bins=30,edgecolor='black')
	plt.title(runway_vector[rwy] + ' <Initial Predicted Start Time - Creation Time> \n ' + str(exempt_window_size) + ' Exempt Window Size')
	#plt.xlim([-1,31])
	plt.tight_layout()

	try:
		plt.figure(11)
		plt.subplot(2,2,rwy+1)
		plt.hist(actual_notice,range=[0,int(math.ceil(max(actual_notice)))],bins=int(math.ceil(max(actual_notice))),edgecolor='black')
		plt.title(runway_vector[rwy] + ' <Actual Start Time - Creation Time> \n ' + str(exempt_window_size) + ' Exempt Window Size')
		#plt.xlim([-1,31])
		plt.tight_layout()
	except:
		pass


plt.figure(10)
plt.savefig('figs/smp_prediction_lead_time_extension_planning_' + str(exempt_window_size) + '_minute_exempt.png')

plt.figure(11)
plt.savefig('figs/smp_actual_lead_time_extension_planning_' + str(exempt_window_size) + '_minute_exempt.png')

plt.show()

	







