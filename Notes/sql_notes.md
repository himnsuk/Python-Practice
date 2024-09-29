SQL query optimization is crucial for improving performance and reducing query execution time. Here are several techniques:

### 1. **Use Indexes Effectively**
   - **Indexes** speed up data retrieval by reducing the amount of data the database must scan. 
   - Create indexes on columns that are frequently used in **WHERE**, **JOIN**, and **GROUP BY** clauses. 
   - However, too many indexes can slow down **INSERT**, **UPDATE**, and **DELETE** operations.

### 2. **Avoid SELECT** \*
   - Instead of selecting all columns with `SELECT *`, explicitly specify the required columns. This reduces the amount of data transferred and processed.

### 3. **Use Appropriate Data Types**
   - Use data types that closely match the data being stored. This reduces memory usage and improves query performance.

### 4. **Use Joins Efficiently**
   - When performing **JOIN** operations, ensure that the columns used for joining are indexed.
   - Avoid **JOIN**ing unnecessary tables or **JOIN**ing large datasets without filters.

### 5. **Filter Early with WHERE**
   - Apply filters early in the query to reduce the number of rows processed. Use the **WHERE** clause to filter data before performing **JOIN**, **GROUP BY**, or **ORDER BY** operations.

### 6. **Optimize Subqueries**
   - Convert subqueries to **JOIN**s if possible. Subqueries can sometimes lead to slower queries because they may require scanning the same data multiple times.
   - Use **EXISTS** instead of **IN** when checking for the existence of data, as **EXISTS** stops scanning once a match is found, making it faster.

### 7. **Use LIMIT for Pagination**
   - For large datasets, use **LIMIT** to retrieve a subset of data instead of retrieving everything at once.

### 8. **Avoid Complex Calculations and Functions in WHERE Clauses**
   - Try to move calculations or functions outside of the **WHERE** clause, as they can prevent the database from using indexes.

### 9. **Use EXPLAIN and ANALYZE**
   - These statements help you understand how the database executes your query. They provide insights into which parts of the query are slow, allowing you to optimize accordingly.

### 10. **Normalize Database Structure**
   - Normalize your database to avoid data duplication, which reduces the amount of data processed during queries. However, excessive normalization may lead to more **JOIN** operations, which can degrade performance.

### 11. **Partition Large Tables**
   - Partitioning divides a large table into smaller parts, which helps in faster querying of large datasets, especially for time-series data.

### 12. **Use Caching**
   - Cache the results of frequently run queries to reduce the database load.

By applying these techniques, you can optimize the performance of your SQL queries significantly.

---
SQL CTE RECURSION
---

```sql
WITH recursive cte_count AS
(
       SELECT 1 n -- Anchor Member
       UNION ALL
       SELECT n + 1 -- Recursive Member
       FROM   cte_count
       WHERE  n < 10 -- Termination Check
)
SELECT n
FROM   cte_count -- Invocation

-- Output
1
2
3
4
5
6
7
8
9
10
```

### Write a query to generate random number between two numbers

```sql
SET @start = 100;SET @end = 200;
WITH recursive cte_count AS
(
       SELECT 1                                       n,
              floor(rand()*(@end-@start+1))+@start AS rand_val -- Anchor Member
       UNION ALL
       SELECT n + 1,
              n + rand_val -- Recursive Member
       FROM   cte_count
       WHERE  n < 10 -- Termination Check
)
SELECT n,
       rand_val
FROM   cte_count -- Invocation
``

