---
id: ezlg7d9hed6b4efcenkg2wu
title: Date_time
desc: ''
updated: 1697459971750
created: 1693659140958
---


### The DATE, DATETIME, and TIMESTAMP Types

1. The DATE type is used for values with a date part but no time part. MySQL retrieves and displays DATE values in 'YYYY-MM-DD' format. The supported range is '1000-01-01' to '9999-12-31'.
2. The DATETIME type is used for values that contain both date and time parts. MySQL retrieves and displays DATETIME values in 'YYYY-MM-DD hh:mm:ss' format. The supported range is '1000-01-01 00:00:00' to '9999-12-31 23:59:59'.
3. The TIMESTAMP data type is used for values that contain both date and time parts. TIMESTAMP has a range of '1970-01-01 00:00:01' UTC to '2038-01-19 03:14:07' UTC.


> A DATETIME or TIMESTAMP value can include a trailing fractional seconds part in up to microseconds (6 digits) precision.In particular, any fractional part in a value inserted into a DATETIME or TIMESTAMP column is stored rather than discarded. With the fractional part included, the format for these values is 'YYYY-MM-DD hh:mm:ss[.fraction]', the range for DATETIME values is '1000-01-01 00:00:00.000000' to '9999-12-31 23:59:59.499999', and the range for TIMESTAMP values is '1970-01-01 00:00:01.000000' to '2038-01-19 03:14:07.499999'. The fractional part should always be separated from the rest of the time by a decimal point; no other fractional seconds delimiter is recognized. For information about fractional seconds support in MySQL

>MySQL converts TIMESTAMP values from the current time zone to UTC for storage, and back from UTC to the current time zone for retrieval. (This does not occur for other types such as DATETIME.) 


| Functions           | Description                                                                                                |
|---------------------|------------------------------------------------------------------------------------------------------------|
| date()              | The date() function is used to get the date from given date/datetime.                                      |
| adddata()           | The adddata() function is used to get the date in which some time/date intervals are added.                |
| curdate()           | The curdate() function is used to get the current date.                                                    |
| current_date()      | The current_date() function is used to get the current date.                                               |
| date_add()          | The date_add() function is used to get the date in which some date/datetime intervals are added.           |
| date_format()       | The date_format() function is used to get the date in specified format.                                    |
| datediff()          | The datediff() function is used to get the difference between the two specified date values.               |
| day()               | The day() function is used to get the day from the given date.                                             |
| dayname()           | The dayname() function is used to get the name of the day from the given date.                             |
| dayofmonth()        | The dayofmonth() function is used to get the day for the specified date.                                   |
| dayofweek()         | The dayofweek() function is used to get the day of the week in numeric.                                    |
| dayofyear()         | The dayofyear() function is used to get the number of day in the year.                                     |
| from_days()         | The from_days() function is used to get the date of the given number of days.                              |
| hour()              | The hour() function is used to get the hour from the given datetime.                                       |
| addtime()           | The addtime() function is used to get the time/datetime value in which some time intervals are added.      |
| current_time()      | The current_time() function is used to get the current time.                                               |
| current_timestamp() | The current_timestamp() function is used to get the current date and time.                                 |
| curtime()           | The curtime() function is used to get the current time.                                                    |
| last_day()          | The last_day() function is used to get the last date of the given month on the date.                       |
| localtime()         | The localtime() function is used to get the current date and time.                                         |
| localtimestamp()    | The localtimestamp() function is used to get the current date and time.                                    |
| makedate()          | The makedate() function is used to make the date from the given year and number of days.                   |
| maketime()          | The maketime() function is used to make the time from given hour, minute and second.                       |
| microsecond()       | The microsecond() function is used to get the value of the microsecond from the given datetime or time.    |
| minute()            | The minute() function is used to get the value of month for the specified datetime or time.                |
| month()             | The month() function is used to get the value of month from given datetime or time.                        |
| monthname()         | The monthname() function is used to get the full month name.                                               |
| now() The now()     | function is used to get the current date and time.                                                         |
| period_add()        | The period_add() function adds the given number of month in the given period in the format YYMM or YYYYMM. |
| period_diff()       | The period_diff() function is used to get the difference between the given two periods.                    |
| quater()            | The quarter() function is used to get the quarter portion of the specified date/datetime.                  |
| sec_to_time()       | The sec_to_time() function is used to convert the specified second into time.                              |
| second()            | The second() function is used to get the second portion from the specified date/datetime.                  |
| str_to_date()       | The str_to_date() function is used to convert the string into the given format_mask.                       |
| subdate()           | The subdate() function is used to get the date which is subtracted by given intervals.                     |
| subtime()           | The subtime() function is used to get the time/datetime which is subtracted by certain intervals.          |
| sysdate()           | The sysdate() function is used to get the system date.                                                     |
| time()              | The time() function is used to get the time for the given time/datetime.                                   |
| time_format()       | The time_format() function is used to format the time in specified format_mask.                            |
| time_to_sec()       | The time_to_sec() function is used to convert the time into seconds.                                       |
| timediff()          | The timediff() function is used to get the difference for the given two time/datetime.                     |
| timestamp()         | The timestamp() function is used to convert the expression into datetime/time.                             |
| to_day()            | The to_day() function is used to convert the date into numeric number of days.                             |
| weekday()           | The weekday() function is used to get the index for a date                                                 |
| week()              | The week() function is used to get the week portion for the specified date.                                |
| weekofyear()        | The weekofyear() function is used to get the week of the given date.                                       |
