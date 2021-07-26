# SQL Interview Preparation
---
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

:::image type="content" source="sql/payment.png" alt-text="payment table":::

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