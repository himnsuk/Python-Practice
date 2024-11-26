
-- Table1 -> Report1

-- Question 1:

-- A company sells 3 products: A, B, and C.table1 contains a sample of order details. Each row in the table is an order. 
-- Orders can either be placed online or over the phone. 
-- Phone orders can be either inbound or outbound. 
-- Outbound phone orders are associated with an agent. 
-- Inbound phone orders do not have an agent. 
-- Online orders don't have this classification and may or may not be associated with an agent.

-- Using the data in table1 produce a report that exactly matches the columns and values in report1.

CREATE TABLE table1 (
    hour INT,
    product VARCHAR(10),
    order_type VARCHAR(20),
    agent VARCHAR(10)
);

INSERT INTO table1 VALUES (0, 'A', 'Online', NULL);
INSERT INTO table1 VALUES (0, 'A', 'Online', NULL);
INSERT INTO table1 VALUES (0, 'B', 'Online', NULL);
INSERT INTO table1 VALUES (0, 'B', 'Phone', NULL);
INSERT INTO table1 VALUES (1, 'A', 'Phone', '947');
INSERT INTO table1 VALUES (1, 'C', 'Online', '947');
INSERT INTO table1 VALUES (2, 'B', 'Phone', '947');
INSERT INTO table1 VALUES (2, 'C', 'Online', NULL);
INSERT INTO table1 VALUES (2, 'C', 'Phone', NULL);
INSERT INTO table1 VALUES (2, 'A', 'Phone', '997');
INSERT INTO table1 VALUES (2, 'C', 'Phone', NULL);

CREATE TABLE report1 (
    hour INT,
    total_orders INT,
    website VARCHAR(30),
    inbound_call VARCHAR(30),
    outbound_call VARCHAR(30),
    distinct_agents INT
);

INSERT INTO report1 VALUES (0, 4, '75.00%', '25.00%', '0.00%', 0);
INSERT INTO report1 VALUES (1, 2, '50.00%', '0.00%', '50.00%', 1);
INSERT INTO report1 VALUES (2, 5, '20.00%', '40.00%', '40.00%', 2);


# ChatGpt solution

SELECT 
    hour,
    COUNT(*) AS total_orders,
    CONCAT(
        ROUND(COUNT(CASE WHEN order_type = 'Online' THEN 1 END) / COUNT(*) * 100, 2),
        '%'
    ) AS website,
    CONCAT(
        ROUND(COUNT(CASE WHEN order_type = 'Phone' AND agent IS NULL THEN 1 END) / COUNT(*) * 100, 2),
        '%'
    ) AS inbound_call,
    CONCAT(
        ROUND(COUNT(CASE WHEN order_type = 'Phone' AND agent IS NOT NULL THEN 1 END) / COUNT(*) * 100, 2),
        '%'
    ) AS outbound_call,
    COUNT(DISTINCT agent) AS distinct_agents
FROM table1
GROUP BY hour
ORDER BY hour;

# MY Solution
with cte1 as (
select hour, count(product) as total_orders, 
sum(CASE WHEN order_type in ("Online") THEN 1 ELSE 0 END) as website,
sum(CASE WHEN order_type in ("Phone") and agent is null THEN 1 ELSE 0 END) as inbound_call,
sum(CASE WHEN order_type in ("Phone") and agent is not null THEN 1 ELSE 0 END) as outbound_call
from table1
group by hour)

select hour, total_orders, 
concat(round((website/total_orders)*100, 2), "%") as website,
concat(round((inbound_call/total_orders)*100, 2), "%") as inbound_call,
concat(round((outbound_call/total_orders)*100, 2), "%") as outbound_call
from cte1