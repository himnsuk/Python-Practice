-- Table2 -> Report2

-- Question 2:

-- table2 contains information regarding digital marketing events. Each row in the table is an independent event. Our clients run advertising campaigns through different channels (search, video, etc.). These ads are served to a group of users, each with a unique user_id. When a user clicks an ad the corresponding client_id and the source (media channel) are captured in this table.

-- At the time of reporting, if the user-client association is based on the same channel (irrespective of the number of events) then it is attributed to that source. If multiple channels are involved, then we display multiple as the source.

-- Using the data in table2 produce a marketing attribution report that exactly matches the columns and values in report2. The report should display users who are associated with just one client and exclude users who are associated with multiple clients.



CREATE TABLE table2 (
    user_id VARCHAR(10),
    client_id VARCHAR(10),
    source VARCHAR(20)
);
INSERT INTO table2 VALUES ('1XesY', '303', 'video');
INSERT INTO table2 VALUES ('1XesY', '303', 'video');
INSERT INTO table2 VALUES ('1XesY', '303', 'video');
INSERT INTO table2 VALUES ('2e55L', '889', 'display');
INSERT INTO table2 VALUES ('2e55L', '889', 'search');
INSERT INTO table2 VALUES ('2e55L', '889', 'email');
INSERT INTO table2 VALUES ('3qyca', '793', 'display');
INSERT INTO table2 VALUES ('3qyca', '474', 'display');
INSERT INTO table2 VALUES ('3qyca', '474', 'display');
INSERT INTO table2 VALUES ('4Et2N', '883', 'search');
INSERT INTO table2 VALUES ('4Et2N', '749', 'email');
INSERT INTO table2 VALUES ('4Et2N', '329', 'email');
INSERT INTO table2 VALUES ('5lRDp', '695', 'search');
INSERT INTO table2 VALUES ('5lRDp', '695', 'search');
INSERT INTO table2 VALUES ('5lRDp', '695', 'search');
INSERT INTO table2 VALUES ('6aOcA', '518', 'display');
INSERT INTO table2 VALUES ('6aOcA', '518', 'video');
INSERT INTO table2 VALUES ('6aOcA', '518', 'video');

CREATE TABLE report2 (
    user_id VARCHAR(10),
    client_id VARCHAR(10),
    source VARCHAR(20)
);

INSERT INTO report2 VALUES ('1XesY', '303', 'video');
INSERT INTO report2 VALUES ('2e55L', '889', 'multiple');
INSERT INTO report2 VALUES ('5lRDp', '695', 'search');
INSERT INTO report2 VALUES ('6aOcA', '518', 'multiple');


-- ChatGpt solution
SELECT 
    user_id,
    CASE 
        WHEN COUNT(DISTINCT client_id) = 1 THEN MAX(client_id)
        ELSE 'multiple'
    END AS client_id,
    CASE 
        WHEN COUNT(DISTINCT source) = 1 THEN MAX(source)
        ELSE 'multiple'
    END AS source
FROM table2
GROUP BY user_id
HAVING COUNT(DISTINCT client_id) = 1
ORDER BY user_id;


-- My Solution

WITH cte
     AS (SELECT DISTINCT *
         FROM   table2)
SELECT user_id,
       Max(client_id) AS client_id,
       CASE
         WHEN Count(source) = 1 THEN Max(source)
         ELSE 'multiple'
       END            AS source
FROM   cte
GROUP  BY user_id
HAVING Count(DISTINCT client_id) = 1 














