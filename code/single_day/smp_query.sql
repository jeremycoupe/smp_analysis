-- Get subset of all of the 
-- metering parameter data records
WITH metering_params AS 
(
	SELECT *
	FROM tmi_resource_metering_values_view
	WHERE record_timestamp BETWEEN '2018-10-02 00:00' AND '2018-10-04 00:00'
		--AND "action" IN ('ADD', 'UPDATE')
	ORDER BY record_id
),
-- Get a subset of all of the SMP records
dmps AS 
(
SELECT 
	record_timestamp,
	id AS dmp_id,
	dmp_status,
	user_action,
	resource_name,
	creation_time,
	start_time,
	end_time,
	flight_count,
	average_gate_hold,
	max_gate_hold,
	action
FROM tmi_departure_metering_program_view
WHERE 
	record_timestamp BETWEEN '2018-10-02 00:00' AND '2018-10-04 00:00'
	--AND "action" IN ('ADD','UPDATE')
ORDER BY record_timestamp
),
-- Join the two subqueries and add a row number
-- that indicates the max metering parameter record
-- with a timestamp less than the SMP record timestamp 
combined AS 
(
SELECT 
	dmps.*,
	metering_params.metering_mode_type,
	metering_params.lead_time_minutes,
	metering_params.time_based_lower_threshold_minutes AS lower_threshold_minutes,
	metering_params.time_based_target_excess_queue_minutes AS target_minutes,
	metering_params.time_based_upper_threshold_minutes AS upper_threshold_minutes,
	metering_params.record_timestamp AS metering_param_timestamp,
	-- When row_num = 1, then it is the record that combines the SMP record with the 
	-- metering parameter record that has the max timestamp less than or equal to the 
	-- timestamp of the SMP record
	ROW_NUMBER() OVER (PARTITION BY dmps.record_timestamp, dmps.resource_name ORDER BY metering_params.record_timestamp DESC) AS row_num 
FROM dmps
JOIN metering_params
ON metering_params.resource_name = dmps.resource_name
	AND metering_params.record_timestamp <= dmps.record_timestamp
)
-- Down select to just the records of interest from the combined
-- results above
SELECT *
FROM combined
WHERE row_num = 1
ORDER BY record_timestamp, resource_name