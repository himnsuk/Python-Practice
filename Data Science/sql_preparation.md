SQL Interview Preparation
---

### Cast Functions and Operators


|Name      |Description                           |
|----------|--------------------------------------|
|BINARY    |   CAST a string to binary string     |
|Row2CAST()|   Cast a value as a certain type     |
|CONVERT() |   Cast a value as a certain type     |


### SQL | WITH clause

- The clause is used for defining a temporary relation such that the output of this temporary relation is available and is used by the query that is associated with the WITH clause.
- Queries that have an associated WITH clause can also be written using nested sub-queries but doing so add more complexity to read/debug the SQL query.
- WITH clause is not supported by all database system.
- The name assigned to the sub-query is treated as though it was an inline view or table
- The SQL WITH clause was introduced by Oracle in the Oracle 9i release 2 database.

```sql
WITH temporaryTable (averageValue) as
    (SELECT avg(Attr1)
    FROM Table)
    SELECT Attr1
    FROM Table
    WHERE Table.Attr1 > temporaryTable.averageValue;
```
Example 1:  Find all the employee whose salary is more than the average salary of all employees. 

Name of the relation: Employee 

<table><tbody><tr><th>EmployeeID</th><th>Name</th><th>Salary</th></tr><tr><th>100011</th><th>Smith</th><th>50000</th></tr><tr><th>100022</th><th>Bill</th><th>94000</th></tr><tr><th>100027</th><th>Sam</th><th>70550</th></tr><tr><th>100845</th><th>Walden</th><th>80000</th></tr><tr><th>115585</th><th>Erik</th><th>60000</th></tr><tr><th>1100070</th><th>Kate</th><th>69000</th></tr></tbody></table>

**SQL Query**
```sql
WITH tempamount(customer_id, totalamount)
     AS (SELECT customer_id,
                Sum(amount)
         FROM   payment
         GROUP  BY customer_id),
     avgamount(averageamount)
     AS (SELECT Avg(amount)
         FROM   payment)
SELECT customer_id,
       totalamount
FROM   tempamount,
       avgamount
WHERE  tempamount.totalamount > avgamount.averageamount 
```

__Output__

<table><tbody><tr><th>EmployeeID</th><th>Name</th><th>Salary</th></tr><tr><th>100022</th><th>Bill</th><th>94000</th></tr><tr><th>100845</th><th>Walden</th><th>80000</th></tr></tbody></table>

**Explanation**: The average salary of all employees is 70591. Therefore, all employees whose salary is more than the obtained average lies in the output relation. 


**Example 2**: Find all the airlines where the total salary of all pilots in that airline is more than the average of total salary of all pilots in the database. 

Name of the relation: **Pilot** 

<table><tbody><tr><th>EmployeeID</th><th>Airline</th><th>Name</th><th>Salary</th></tr><tr><th>70007</th><th>Airbus 380</th><th>Kim</th><th>60000</th></tr><tr><th>70002</th><th>Boeing</th><th>Laura</th><th>20000</th></tr><tr><th>10027</th><th>Airbus 380</th><th>Will</th><th>80050</th></tr><tr><th>10778</th><th>Airbus 380</th><th>Warren</th><th>80780</th></tr><tr><th>115585</th><th>Boeing</th><th>Smith</th><th>25000</th></tr><tr><th>114070</th><th>Airbus 380</th><th>Katy</th><th>78000</th></tr></tbody></table>


**SQL Query:**

```sql
WITH tempamount(customer_id, totalamount)
     AS (SELECT customer_id,
                Sum(amount)
         FROM   payment
         GROUP  BY customer_id),
     avgamount(averageamount)
     AS (SELECT Avg(amount)
         FROM   payment)
SELECT customer_id,
       totalamount
FROM   tempamount,
       avgamount
WHERE  tempamount.totalamount > avgamount.averageamount 
```

**Question**: Count distinct on two columns


![Payment Table](sql/payment.png)

```sql
WITH distinctcustid(distinctcustomercount)
     AS (SELECT Count(DISTINCT( customer_id ))
         FROM   sakila.payment),
     distinctstaffid(distinctstaffcount)
     AS (SELECT Count(DISTINCT( staff_id ))
         FROM   sakila.payment)
SELECT distinctcustomercount,
       distinctstaffcount
FROM   distinctcustid,
       distinctstaffid 
```

**Question**: Difference between _UNION_ and _UNIOINALL_ ?

The only difference between Union and Union All is that Union extracts the rows that are being specified in the query while Union All extracts all the rows including the duplicates (repeated values) from both the queries


### Window Functions

**Types of Window functions**
- Aggregate Window Functions. SUM(), MAX(), MIN(), AVG(). COUNT()
- Ranking Window Functions. RANK(), DENSE_RANK(), ROW_NUMBER(), NTILE()
- Value Window Functions. LAG(), LEAD(), FIRST_VALUE(), LAST_VALUE()


```sql

window_function ( [ ALL ] expression ) 
OVER ( [ PARTITION BY partition_list ] [ ORDER BY order_list] )
```

#### Arguments

**window_function**

Specify the name of the window function

**ALL**

ALL is an optional keyword. When you will include ALL it will count all values including duplicate ones. DISTINCT is not supported in window functions

**expression**

The target column or expression that the functions operates on. In other words, the name of the column for which we need an aggregated value. For example, a column containing order amount so that we can see total orders received.

**OVER**

Specifies the window clauses for aggregate functions.

**PARTITION BY partition_list**

Defines the window (set of rows on which window function operates) for window functions. We need to provide a field or list of fields for the partition after PARTITION BY clause. Multiple fields need be separated by a comma as usual. If PARTITION BY is not specified, grouping will be done on entire table and values will be aggregated accordingly.

**ORDER BY order_list**
Sorts the rows within each partition. If ORDER BY is not specified, ORDER BY uses the entire table.


#### Example

Create a table

```sql
CREATE TABLE Orders2
(
	order_id INT,
	order_date DATE,
	customer_name VARCHAR(250),
	city VARCHAR(100),	
	order_amount DECIMAL
);
 
INSERT INTO Orders2 (
	order_id ,
	order_date ,
	customer_name,
	city ,	
	order_amount
)
SELECT '1002',STR_TO_DATE('04-02-2017', '%m-%d-%Y'),'David Jones','Arlington',20000
UNION ALL	  
SELECT '1003',STR_TO_DATE('04-03-2017', '%m-%d-%Y'),'John Smith','Shalford',5000
UNION ALL	  
SELECT '1004',STR_TO_DATE('04-04-2017', '%m-%d-%Y'),'Michael Smith','GuildFord',15000
UNION ALL	  
SELECT '1005',STR_TO_DATE('04-05-2017', '%m-%d-%Y'),'David Williams','Shalford',7000
UNION ALL	  
SELECT '1006',STR_TO_DATE('04-06-2017', '%m-%d-%Y'),'Paum Smith','GuildFord',25000
UNION ALL	 
SELECT '1007',STR_TO_DATE('04-10-2017', '%m-%d-%Y'),'Andrew Smith','Arlington',15000
UNION ALL	  
SELECT '1008',STR_TO_DATE('04-11-2017', '%m-%d-%Y'),'David Brown','Arlington',2000
UNION ALL	  
SELECT '1009',STR_TO_DATE('04-20-2017', '%m-%d-%Y'),'Robert Smith','Shalford',1000
UNION ALL	  
SELECT '1010',STR_TO_DATE('04-25-2017', '%m-%d-%Y'),'Peter Smith','GuildFord',500
```

#### Aggregate Window Functions

**SUM()**

Order Table


![Order Table ](sql/OrderTable.png)

**Query** Get sum of _order\_amount_ per city

```sql
SELECT *,
       Sum(order_amount) OVER(PARTITION BY city) AS total_amount
FROM   orders2 
```

**Output**

![Sum Over City](sql/sum_over_city.png)

> Same type of syntax can be used for _AVG()_, _MIN()_, _MAX()_, _COUNT()_

**AVG()**

```sql
SELECT *,
       AVG(order_amount) OVER(PARTITION BY city) AS avg_amount
FROM   orders2 
```

![AVG OVER CITY](sql/avg_over_city.png)

**MIN()**

```sql
SELECT *,
       MIN(order_amount) OVER(PARTITION BY city) AS min_amount
FROM   orders2 
```

![MIN OVER CITY](sql/min_over_city.png)

**MAX()**

```sql
SELECT *,
       MAX(order_amount) OVER(PARTITION BY city) AS max_amount
FROM   orders2 
```

![MAX OVER CITY](sql/max_over_city.png)

**COUNT()**

```sql
SELECT *, COUNT(order_id) OVER(PARTITION BY city) as total_id
FROM orders2
```

![COUNT OVER CITY](sql/count_over_city.png)

### Ranking Functions

**RANK(), DENSE_RANK(), ROW_NUMBER(), NTILE()**


**RANK()**

The RANK() function is used to give a unique rank to each record based on a specified value, for example salary, order amount etc.

If two records have the same value then the RANK() function will assign the same rank to both records by skipping the next rank. This means – if there are two identical values at rank 2, it will assign the same rank 2 to both records and then skip rank 3 and assign rank 4 to the next record.

Let’s rank each order by their order amount.

```sql
SELECT order_id, order_date, customer_name, city,
	RANK() OVER(ORDER BY order_amount DESC) ranked_items
FROM orders2
```


![Rank Function](sql/rank.png)

**DENSE_RANK()**

```sql
SELECT order_id, order_date, customer_name, city,
	DENSE_RANK() OVER(ORDER BY order_amount DESC) ranked_items
FROM orders2
```
![Dense Rank Function](sql/dense_rank.png)


_DENSE\_RANK()_ over city

```sql
SELECT order_id, order_date, customer_name, city,
	DENSE_RANK() OVER(PARTITION BY city ORDER BY order_amount DESC) ranked_items
FROM orders2;
```


![Dense Rank Over City](sql/rank_over_city.png)


**ROW_NUMBER**

```sql
SELECT order_id, order_date, customer_name, city,
	ROW_NUMBER() OVER(ORDER BY order_amount DESC) `row_number`
FROM orders2
```


![ROW_NUMBER Function](sql/row_number.png)


**NTILE()**

NTILE() is a very helpful window function. It helps you to identify what percentile (or quartile, or any other subdivision) a given row falls into.

This means that if you have 100 rows and you want to create 4 quartiles based on a specified value field you can do so easily and see how many rows fall into each quartile.

Let’s see an example. In the query below, we have specified that we want to create four quartiles based on order amount. We then want to see how many orders fall into each quartile.


```sql
SELECT order_id, order_date, customer_name, city,
	NTILE(4) OVER(ORDER BY order_amount DESC) ranked_items
FROM orders2;
```

![NTILE Function](sql/ntile.png)


#### Value Window Functions

Value window functions are used to find first, last, previous and next values. The functions that can be used are *LAG(), LEAD(), FIRST_VALUE(), LAST_VALUE()*

**LAG()**

```sql
SELECT order_id, customer_name, city,order_date,
	LAG(order_date, 1) OVER(ORDER BY order_amount DESC) `LAG`
FROM orders2;
```

![LAG Function](sql/lag.png)


**LEAD()**

```sql
SELECT order_id, customer_name, city, order_date,
	LEAD(order_date, 1) OVER(ORDER BY order_amount DESC) `LEAD`
FROM orders2;
```

![LEAD() Function](sql/lead.png)


**FIRST_VALUE()**

```sql
SELECT order_id, order_date, customer_name, city,
	FIRST_VALUE(order_date) OVER(PARTITION BY city ORDER BY city) `FIRST_VALUE`
FROM orders2;
```

![FIRST_VALUE() Function](sql/first_vallue.png)

**LAST_VALUE()**

```sql
SELECT order_id, order_date, customer_name, city,
	LAST_VALUE(order_date) OVER(PARTITION BY city ORDER BY city) `LAST_VALUE`
FROM orders2
```


![LAST_VALUE() Function](sql/last_value.png)


### Common Table Expression

A Common Table Expression, also called as CTE in short form, is a temporary named result set that you can reference within a SELECT, INSERT, UPDATE, or DELETE statement. The CTE can also be used in a View.

**Difference between _nested query_ and _CTE_**

In the sub-query vs simple (non-recursive) CTE versions, they are probably very similar. You would have to use the profiler and actual execution plan to spot any differences, and that would be specific to your setup (so we can't tell you the answer in full).

In general; A CTE can be used recursively; a sub-query cannot. This makes them especially well suited to tree structures.

**Syntax**

```sql
WITH expression_name [ ( column_name [,...n] ) ] 
AS 
( CTE_query_definition )
```

**To view the CTE result we use a Select query with the CTE expression name.**

```sql
Select [Column1,Column2,Column3 …..] from expression_name
```

or

```sql
Select * from expression_name
```

#### Common Table Expression (CTE) Types

**Non-Recursive CTEs**

Non-Recursive CTEs are simple where the CTE doesn’t use any recursion, or repeated processing in of a sub-routine. We will create a simple Non-Recursive CTE to display the row number from 1 to 10.

As per the CTE Syntax each CTE query will start with a “With” followed by the CTE Expression name with column list.

Here we have been using only one column as ROWNO. Next is the Query part, here we write our select query to be execute for our CTE. After creating our CTE query to run the CTE use the select statement with CTE Expression name.

```sql
;with ROWCTE(ROWNO) as  
   (  
     SELECT 
  ROW_NUMBER() OVER(ORDER BY name ASC) AS ROWNO
FROM sys.databases 
WHERE database_id <= 10
    )  
 
SELECT * FROM ROWCTE 
```



## CASE Statement

**Searched CASE in SELECT (Matches a boolean expression)**

```sql
SELECT Id, itemId, Price,
    CASE WHEN Price < 10 THEN 'CHEAP'
         WHEN Price < 20 THEN 'AFFORDABLE'
         ELSE 'EXPENSIVE'
    END AS PriceRating
FROM ItemSales
```

**Use CASE to COUNT the number of rows in a column match a condition.**

```sql
SELECT
    COUNT(ID) AS ItemsCount,
    SUM(CASE
            WHEN PriceRating = 'Expensive' THEN 1
            ELSE 0
        END
        ) AS ExpensiveItemsCount
FROM ItemSales
```


**Shorthand CASE in SELECT**

Shorthand variant evaluates an expression (usually a column) against a series of values.
This variant is a bit shsorter, and saves repeating evaluated expression over and over again.
The _ELSE_ clause can still be used though:

```sql
SELECT Id, ItemId, Price,
    CASE Price WHEN 5 THEN 'CHEAP'
               WHEN 15 THEN 'AFFORDABLE'
               ELSE         'EXPENSIVE'
    END as PriceRating
FROM ItemSales
```

**CASE in a clause ORDER BY**

```sql
SELECT * FROM DEPT
ORDER BY
    CASE DEPARTMENT
        WHEN 'MARKETING' THEN 1
        WHEN 'SALES' THEN 2
        WHEN 'RESEARCH' THEN 3
        WHEN 'INNOVATION' THEN 4
        ELSE 5
    END, CITY
```

**Using CASE in UPDATE**

```sql
UPDATE ItemPrice
SET Price = Price *
    CASE ItemId
        WHENN 1 THEN 1.05
        WHENN 2 THEN 1.10
        WHENN 3 THEN 1.15
        ELSE 1.00
    END
```

**Recursive CTE**

```sql
With
  EmployeesCTE (EmployeeId, Name, ManagerId, [Level])
  as
  (
    Select EmployeeId, Name, ManagerId, 1
    from tblEmployee
    where ManagerId is null
    
    union all
    
    Select tblEmployee.EmployeeId, tblEmployee.Name, 
    tblEmployee.ManagerId, EmployeesCTE.[Level] + 1
    from tblEmployee
    join EmployeesCTE
    on tblEmployee.ManagerID = EmployeesCTE.EmployeeId
  )
Select EmpCTE.Name as Employee, Isnull(MgrCTE.Name, 'Super Boss') as Manager, 
EmpCTE.[Level] 
from EmployeesCTE EmpCTE
left join EmployeesCTE MgrCTE
on EmpCTE.ManagerId = MgrCTE.EmployeeId
```

OR

```sql
WITH RECURSIVE ManagedByJames(Level, ID, FName, LName) AS(
    --Start with this row
    SELECT 1, ID, FName, LName
    FROM Employees
    WHERE ID = 1

    UNION ALL
    -- get employees that have any of the previously selected rows as manager
    SELECT ManagedByJames.Level + 1,
           Employees.ID,
           Employees.FName,
           Employees.LName
    FROM Employees
    JOIN ManagedByJames
        ON Employees.ManagerID = ManagedByJames.ID
    ORDER BY 1 DESC -- depth-first search
)
SELECT * FROM ManagedByJames;
```


## Create a new function

CREATE FUNCTION create as user-defined function that can be used when doing a SELECT, INSERT, UPDATE, or DELETE query. The functions can be created to return a single variable or a single table

Examples

Create a new Function

```sql
CREATE FUNCTION FirstWord(@input varchar(1000))
RETURNS varchar(1000)
AS
BEGIN
    DECLARE @outut varchar(10000)
    SET @output = SUBSTRING(@input, 0, CASE CHARINDEX(' ', @input))
        WHEN 0 THEN LEN(@input) + 1
        ELSE CHARINDEX(' ', @input)
    END)
    RETURN @output
END

```


1. CUME_DIST (Transact-SQL)
1. FIRST_VALUE (Transact-SQL)
1. LAG (Transact-SQL)
1. LAST_VALUE (Transact-SQL)
1. LEAD (Transact-SQL)
1. PERCENT_RANK (Transact-SQL)
1. PERCENTILE_CONT (Transact-SQL)
1. PERCENTILE_DISC (Transact-SQL)



**Useful aggregate functions:**

1. AVG() - Returns the average value
1. COUNT() - Returns the number of rows
1. FIRST() - Returns the first value
1. LAST() - Returns the last value
1. MAX() - Returns the largest value
1. MIN() - Returns the smallest value
1. SUM() - Returns the sum


**Useful scalar functions:**

1. UCASE() - Converts a field to upper case
1. LCASE() - Converts a field to lower case
1. MID() - Extract characters from a text field
1. LEN() - Returns the length of a text field
1. ROUND() - Rounds a numeric field to the number of decimals specified
1. NOW() - Returns the current system date and time
1. FORMAT() - Formats how a field is to be displayed


1. CAST
2. CONVERT
3. PARSE
4. DATENAME
5. GETDATE()
6. DATEADD()
7. CHOOSE
8.  IIF
9. SIGN
10. POWER


### Materialized View

A Materialized View persists the data returned from the view definition query and automatically gets updated as data changes in the underlying tables. It improves the performance of complex queries (typically queries with joins and aggregations) while offering simple maintenance operations. With its execution plan automatching capability, a materialized view does not have to be referenced in the query for the optimizer to consider the view for substitution. This capability allows data engineers to implement materialized views as a mechanism for improving query response time, without having to change queries.

**Syntax:**

```sql
CREATE MATERIALIZED VIEW [ schema_name. ] materialized_view_name
    WITH (  
      <distribution_option>
    )
    AS <select_statement>
[;]

<distribution_option> ::=
    {  
        DISTRIBUTION = HASH ( distribution_column_name )  
      | DISTRIBUTION = ROUND_ROBIN  
    }

<select_statement> ::=
    SELECT select_criteria
```



### Execution Order

![Execution Order](sql/execution_order_sql.png)

---
Difference b/w CTE and Temp Tables
---

Both **CTEs (Common Table Expressions)** and **Temp Tables** are used in SQL to store temporary result sets, but they have key differences in terms of usage, scope, and performance. Here's a breakdown to help you understand and remember the differences:

---

### 1. **CTE (Common Table Expression)**
   - **Definition**: A CTE is a temporary result set that exists only during the execution of a single query. It is defined using the `WITH` clause.
   - **Syntax**:
     ```sql
     WITH cte_name AS (
         SELECT column1, column2
         FROM table_name
         WHERE condition
     )
     SELECT * FROM cte_name;
     ```
   - **Scope**: 
     - A CTE is only accessible within the query where it is defined.
     - It cannot be reused in other queries or sessions.
   - **Lifetime**:
     - A CTE exists only for the duration of the query execution.
     - Once the query completes, the CTE is discarded.
   - **Performance**:
     - CTEs are not stored physically in the database.
     - They are more like inline views and are optimized as part of the query execution plan.
   - **Use Cases**:
     - Simplifying complex queries by breaking them into smaller, more readable parts.
     - Recursive queries (e.g., hierarchical data like organizational charts).

---

### 2. **Temp Table**
   - **Definition**: A temp table is a physical table that is created temporarily in the database and can store data for the duration of a session or transaction.
   - **Syntax**:
     ```sql
     CREATE TEMPORARY TABLE temp_table_name AS (
         SELECT column1, column2
         FROM table_name
         WHERE condition
     );
     ```
   - **Scope**:
     - A temp table is accessible within the session or transaction where it is created.
     - It can be reused across multiple queries in the same session.
   - **Lifetime**:
     - A temp table exists until the session ends or it is explicitly dropped.
     - In some databases (e.g., SQL Server), temp tables are automatically dropped when the session ends.
   - **Performance**:
     - Temp tables are stored physically in the database (usually in a temporary storage space like `tempdb` in SQL Server).
     - They can have indexes, statistics, and constraints, which can improve performance for complex queries.
   - **Use Cases**:
     - Storing intermediate results for reuse across multiple queries.
     - Handling large datasets that require indexing or partitioning.
     - Performing batch processing or ETL operations.

---

### Key Differences Between CTEs and Temp Tables

| Feature                | CTE                              | Temp Table                      |
|------------------------|----------------------------------|---------------------------------|
| **Scope**              | Limited to the query where it is defined | Accessible across multiple queries in the same session |
| **Lifetime**           | Exists only during query execution | Exists until the session ends or it is explicitly dropped |
| **Storage**            | Not stored physically            | Stored physically in the database (e.g., `tempdb`) |
| **Performance**        | Optimized as part of the query   | Can have indexes, statistics, and constraints for better performance |
| **Reusability**        | Cannot be reused outside the query | Can be reused across multiple queries |
| **Use Cases**          | Simplifying complex queries, recursive queries | Storing intermediate results, batch processing |

---

### When to Use CTEs vs. Temp Tables
- **Use CTEs**:
  - When you need a temporary result set for a single query.
  - When you want to simplify complex queries or perform recursive operations.
  - When you don’t need to reuse the result set across multiple queries.

- **Use Temp Tables**:
  - When you need to store intermediate results for reuse across multiple queries.
  - When you are working with large datasets and need indexing or partitioning for performance.
  - When you need to persist data temporarily for the duration of a session.

---

### Pro Tip to Remember
- Think of a **CTE** as a **disposable view** that exists only for the duration of a single query.
- Think of a **Temp Table** as a **physical table** that exists temporarily in the database and can be reused.

By understanding these differences and practicing with real-world examples, you'll never confuse CTEs and temp tables again!

---
Aggregate function list
---

SQL aggregate functions are used to perform calculations on a set of values and return a single value. They are commonly used in `GROUP BY` clauses to summarize data. Here's a list of the most commonly used SQL aggregate functions:

---

### 1. **COUNT**
   - **Purpose**: Counts the number of rows in a result set.
   - **Syntax**:
     ```sql
     COUNT(column_name)
     ```
   - **Example**:
     ```sql
     SELECT COUNT(*) FROM employees; -- Counts all rows
     SELECT COUNT(department_id) FROM employees; -- Counts non-null values in department_id
     ```

---

### 2. **SUM**
   - **Purpose**: Calculates the sum of numeric values in a column.
   - **Syntax**:
     ```sql
     SUM(column_name)
     ```
   - **Example**:
     ```sql
     SELECT SUM(salary) FROM employees; -- Sum of all salaries
     ```

---

### 3. **AVG**
   - **Purpose**: Calculates the average of numeric values in a column.
   - **Syntax**:
     ```sql
     AVG(column_name)
     ```
   - **Example**:
     ```sql
     SELECT AVG(salary) FROM employees; -- Average salary
     ```

---

### 4. **MIN**
   - **Purpose**: Finds the minimum value in a column.
   - **Syntax**:
     ```sql
     MIN(column_name)
     ```
   - **Example**:
     ```sql
     SELECT MIN(salary) FROM employees; -- Minimum salary
     ```

---

### 5. **MAX**
   - **Purpose**: Finds the maximum value in a column.
   - **Syntax**:
     ```sql
     MAX(column_name)
     ```
   - **Example**:
     ```sql
     SELECT MAX(salary) FROM employees; -- Maximum salary
     ```

---

### 6. **GROUP_CONCAT** (MySQL) / **STRING_AGG** (SQL Server, PostgreSQL)
   - **Purpose**: Concatenates values from multiple rows into a single string.
   - **Syntax**:
     ```sql
     GROUP_CONCAT(column_name SEPARATOR ', ') -- MySQL
     STRING_AGG(column_name, ', ') -- SQL Server, PostgreSQL
     ```
   - **Example**:
     ```sql
     SELECT department_id, GROUP_CONCAT(employee_name SEPARATOR ', ') 
     FROM employees 
     GROUP BY department_id; -- Concatenates employee names by department
     ```

---

### 7. **VAR_POP** / **VAR_SAMP**
   - **Purpose**: Calculates the population variance (`VAR_POP`) or sample variance (`VAR_SAMP`) of a set of values.
   - **Syntax**:
     ```sql
     VAR_POP(column_name)
     VAR_SAMP(column_name)
     ```
   - **Example**:
     ```sql
     SELECT VAR_POP(salary) FROM employees; -- Population variance of salaries
     SELECT VAR_SAMP(salary) FROM employees; -- Sample variance of salaries
     ```

---

### 8. **STDDEV_POP** / **STDDEV_SAMP**
   - **Purpose**: Calculates the population standard deviation (`STDDEV_POP`) or sample standard deviation (`STDDEV_SAMP`) of a set of values.
   - **Syntax**:
     ```sql
     STDDEV_POP(column_name)
     STDDEV_SAMP(column_name)
     ```
   - **Example**:
     ```sql
     SELECT STDDEV_POP(salary) FROM employees; -- Population standard deviation of salaries
     SELECT STDDEV_SAMP(salary) FROM employees; -- Sample standard deviation of salaries
     ```

---

### 9. **ARRAY_AGG** (PostgreSQL)
   - **Purpose**: Aggregates values into an array.
   - **Syntax**:
     ```sql
     ARRAY_AGG(column_name)
     ```
   - **Example**:
     ```sql
     SELECT department_id, ARRAY_AGG(employee_name) 
     FROM employees 
     GROUP BY department_id; -- Aggregates employee names into arrays by department
     ```

---

### 10. **JSON_ARRAYAGG** (MySQL, PostgreSQL)
   - **Purpose**: Aggregates values into a JSON array.
   - **Syntax**:
     ```sql
     JSON_ARRAYAGG(column_name)
     ```
   - **Example**:
     ```sql
     SELECT department_id, JSON_ARRAYAGG(employee_name) 
     FROM employees 
     GROUP BY department_id; -- Aggregates employee names into JSON arrays by department
     ```

---

### 11. **BIT_AND** / **BIT_OR** / **BIT_XOR**
   - **Purpose**: Performs bitwise operations on a set of values.
   - **Syntax**:
     ```sql
     BIT_AND(column_name)
     BIT_OR(column_name)
     BIT_XOR(column_name)
     ```
   - **Example**:
     ```sql
     SELECT BIT_AND(flags) FROM permissions; -- Bitwise AND of flags
     SELECT BIT_OR(flags) FROM permissions; -- Bitwise OR of flags
     SELECT BIT_XOR(flags) FROM permissions; -- Bitwise XOR of flags
     ```

---

### 12. **APPROX_COUNT_DISTINCT** (BigQuery, Spark SQL)
   - **Purpose**: Provides an approximate count of distinct values in a column (useful for large datasets).
   - **Syntax**:
     ```sql
     APPROX_COUNT_DISTINCT(column_name)
     ```
   - **Example**:
     ```sql
     SELECT APPROX_COUNT_DISTINCT(user_id) FROM logs; -- Approximate distinct user IDs
     ```

---

### Summary Table of Aggregate Functions

| Function               | Purpose                                      |
|------------------------|----------------------------------------------|
| `COUNT()`              | Counts rows or non-null values               |
| `SUM()`                | Sums numeric values                          |
| `AVG()`                | Calculates the average of numeric values     |
| `MIN()`                | Finds the minimum value                     |
| `MAX()`                | Finds the maximum value                     |
| `GROUP_CONCAT()`       | Concatenates values into a string (MySQL)    |
| `STRING_AGG()`         | Concatenates values into a string (SQL Server, PostgreSQL) |
| `VAR_POP()` / `VAR_SAMP()` | Calculates variance                        |
| `STDDEV_POP()` / `STDDEV_SAMP()` | Calculates standard deviation       |
| `ARRAY_AGG()`          | Aggregates values into an array (PostgreSQL) |
| `JSON_ARRAYAGG()`      | Aggregates values into a JSON array          |
| `BIT_AND()` / `BIT_OR()` / `BIT_XOR()` | Performs bitwise operations       |
| `APPROX_COUNT_DISTINCT()` | Approximate count of distinct values      |

---

### Pro Tip
- Aggregate functions are often used with the `GROUP BY` clause to group data before applying the function.
- Use `DISTINCT` inside aggregate functions (e.g., `COUNT(DISTINCT column_name)`) to count or calculate unique values.

By understanding these functions and practicing with real-world examples, you'll master SQL aggregation!

---
Logical Order of execution
---

Understanding the **logical order of SQL query processing** is crucial for writing efficient and accurate queries. The logical order determines how SQL evaluates and executes different parts of a query. Here's a breakdown of the steps in a way that's easy to remember:

---

### Logical Order of SQL Query Processing

1. **FROM** (and JOINs)
2. **WHERE**
3. **GROUP BY**
4. **HAVING**
5. **SELECT**
6. **DISTINCT**
7. **ORDER BY**
8. **LIMIT** / **OFFSET**

---

### Step-by-Step Explanation

#### 1. **FROM (and JOINs)**
   - **What Happens**: The query starts by identifying the tables and joining them (if necessary).
   - **Why It Matters**: This step creates a working dataset by combining rows from one or more tables.
   - **Example**:
     ```sql
     FROM employees
     JOIN departments ON employees.department_id = departments.id
     ```

#### 2. **WHERE**
   - **What Happens**: Filters rows based on specified conditions.
   - **Why It Matters**: Reduces the dataset by removing rows that don't meet the criteria.
   - **Example**:
     ```sql
     WHERE employees.salary > 50000
     ```

#### 3. **GROUP BY**
   - **What Happens**: Groups rows that have the same values in specified columns.
   - **Why It Matters**: Prepares the data for aggregation (e.g., `COUNT`, `SUM`, `AVG`).
   - **Example**:
     ```sql
     GROUP BY department_id
     ```

#### 4. **HAVING**
   - **What Happens**: Filters groups based on conditions (used after `GROUP BY`).
   - **Why It Matters**: Allows filtering of aggregated results.
   - **Example**:
     ```sql
     HAVING COUNT(employees.id) > 10
     ```

#### 5. **SELECT**
   - **What Happens**: Selects the columns to include in the final result set.
   - **Why It Matters**: Determines what data is returned.
   - **Example**:
     ```sql
     SELECT department_id, AVG(salary) AS avg_salary
     ```

#### 6. **DISTINCT**
   - **What Happens**: Removes duplicate rows from the result set.
   - **Why It Matters**: Ensures unique values in the output.
   - **Example**:
     ```sql
     SELECT DISTINCT department_id
     ```

#### 7. **ORDER BY**
   - **What Happens**: Sorts the result set by specified columns.
   - **Why It Matters**: Organizes the output in a meaningful way.
   - **Example**:
     ```sql
     ORDER BY avg_salary DESC
     ```

#### 8. **LIMIT / OFFSET**
   - **What Happens**: Limits the number of rows returned and skips a specified number of rows.
   - **Why It Matters**: Controls pagination or restricts the size of the result set.
   - **Example**:
     ```sql
     LIMIT 10 OFFSET 20
     ```

---

### Visual Representation
To remember the order, think of it as a **funnel** that narrows down the data step by step:

```
FROM → WHERE → GROUP BY → HAVING → SELECT → DISTINCT → ORDER BY → LIMIT/OFFSET
```

---

### Pro Tip: Use a Mnemonic
To make it even easier to remember, use this mnemonic:
**"Frogs With Green Hats Sing Delightful Odes Loudly."**

- **F**rogs → **F**ROM
- **W**ith → **W**HERE
- **G**reen → **G**ROUP BY
- **H**ats → **H**AVING
- **S**ing → **S**ELECT
- **D**elightful → **D**ISTINCT
- **O**des → **O**RDER BY
- **L**oudly → **L**IMIT/OFFSET

---

### Why This Order Matters
- **Efficiency**: Understanding the order helps you write queries that minimize unnecessary processing (e.g., filtering early with `WHERE`).
- **Accuracy**: Ensures you apply conditions and transformations in the correct sequence.
- **Debugging**: Helps you identify why a query might not be returning the expected results.

---

### Example Query with Logical Order
```sql
SELECT department_id, AVG(salary) AS avg_salary
FROM employees
JOIN departments ON employees.department_id = departments.id
WHERE employees.salary > 50000
GROUP BY department_id
HAVING COUNT(employees.id) > 10
ORDER BY avg_salary DESC
LIMIT 5;
```

---

By understanding the logical order and using the mnemonic, you'll never forget how SQL processes queries!