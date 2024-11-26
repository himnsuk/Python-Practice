Here are 10 frequently asked database design questions that are often seen in interviews, especially for roles like Data Analyst, Data Engineer, or Database Administrator. Each question focuses on designing a schema for common business use cases and testing your understanding of data relationships, indexing, and normalization.

### 1. **Design a Database for a Social Media Platform**
   - Define tables for **users**, **posts**, **comments**, **likes**, **follows**, and **messages**.
   - Consider relationships for friend connections, message threads, and tracking reactions on posts.
   - Think about indexing for fast retrieval of posts, followers, and comments for each user.

### 2. **Create a Schema for an E-commerce Website**
   - Define tables for **customers**, **products**, **orders**, **order_items**, **payments**, and **reviews**.
   - Address relationships between customers and orders, products in orders, and the need for transaction history.
   - Consider indexing on customer and product IDs for quick lookup, and designing a schema that can support promotions and discounts.

### 3. **Database for a Hospital Management System**
   - Define tables for **patients**, **doctors**, **appointments**, **treatments**, **prescriptions**, and **medical records**.
   - Handle relationships between doctors and patients, patients and treatments, and maintaining patient history.
   - Consider indexing on appointment dates and patient IDs for efficient searching.

### 4. **Design a Schema for a Library Management System**
   - Define tables for **books**, **members**, **loans**, **authors**, and **categories**.
   - Model the relationships between books and authors, books and members (for borrowing), and tracking due dates.
   - Optimize with indexes on book title, author, and due date for retrieval.

### 5. **Build a Database for a Food Delivery Service**
   - Define tables for **restaurants**, **customers**, **orders**, **order_items**, **delivery_agents**, and **payments**.
   - Focus on relationships between restaurants and menu items, orders and delivery agents, and the customer’s order history.
   - Add indexes on restaurant ID, customer ID, and order status for faster searches.

### 6. **Database Design for a Banking System**
   - Define tables for **customers**, **accounts**, **transactions**, **loans**, and **branches**.
   - Focus on relationships between customers and accounts, account types (savings, checking), and tracking transactions.
   - Consider indexing on account numbers and transaction timestamps for quick lookups and efficient reporting.

### 7. **Schema for a Movie Streaming Platform**
   - Define tables for **users**, **movies**, **subscriptions**, **watch_history**, **ratings**, and **genres**.
   - Model the relationship between users and their subscriptions, movie genres, and viewing history.
   - Index popular columns like user ID, movie ID, and rating to support efficient querying for recommendations and analytics.

### 8. **Design a Database for a School Management System**
   - Define tables for **students**, **teachers**, **courses**, **enrollments**, **grades**, and **attendance**.
   - Model relationships between students and courses, teachers and courses, and track each student's grades.
   - Index on student IDs and course codes to allow fast access to grades, attendance records, and enrollments.

### 9. **Database for an Online Booking System (e.g., for flights or hotels)**
   - Define tables for **users**, **bookings**, **flights** or **rooms**, **payment_details**, and **locations**.
   - Address relationships for each user’s booking history, payment information, and flight or room details.
   - Use indexes on booking ID, user ID, and dates for faster searching and filtering of booking history.

### 10. **Design a Database for a Fitness Tracking App**
   - Define tables for **users**, **workouts**, **exercises**, **goals**, and **progress_logs**.
   - Focus on relationships between users and their workout routines, exercises within workouts, and tracking progress over time.
   - Index user ID, date of workout, and goal to allow efficient retrieval of workout histories and goal tracking.

### Tips for Answering Database Design Questions
   - **Identify Core Entities and Relationships**: Clearly define the main entities and how they relate (e.g., one-to-many, many-to-many).
   - **Normalization and Redundancy**: Consider normalization to avoid data redundancy and ensure data integrity.
   - **Indexing Strategy**: Choose key columns to index based on query needs, such as unique IDs, foreign keys, or frequently filtered columns.
   - **Scalability and Performance**: Consider the volume of data and design with scalability in mind.