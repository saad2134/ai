-- AI-Powered Product Recommendations using SQL
-- Simple collaborative filtering approach

-- Create sample e-commerce data
CREATE TABLE users (user_id INT, user_name VARCHAR(50));
CREATE TABLE products (product_id INT, product_name VARCHAR(50), category VARCHAR(50));
CREATE TABLE purchases (user_id INT, product_id INT, rating INT, purchase_date DATE);

-- Insert sample data
INSERT INTO users VALUES 
(1, 'Alice'), (2, 'Bob'), (3, 'Charlie'), (4, 'Diana');

INSERT INTO products VALUES
(1, 'Laptop', 'Electronics'), (2, 'Book', 'Education'),
(3, 'Headphones', 'Electronics'), (4, 'Notebook', 'Education');

INSERT INTO purchases VALUES
(1, 1, 5, '2024-01-01'), (1, 2, 4, '2024-01-02'),
(2, 1, 4, '2024-01-01'), (2, 3, 5, '2024-01-03'),
(3, 2, 5, '2024-01-02'), (3, 4, 3, '2024-01-04');

-- Basic Recommendation: "Users who bought this also bought..."
SELECT 
    p1.product_name AS purchased_item,
    p2.product_name AS recommended_item,
    COUNT(*) AS frequency
FROM purchases pur1
JOIN purchases pur2 ON pur1.user_id = pur2.user_id 
    AND pur1.product_id != pur2.product_id
JOIN products p1 ON pur1.product_id = p1.product_id
JOIN products p2 ON pur2.product_id = p2.product_id
GROUP BY p1.product_name, p2.product_name
ORDER BY frequency DESC;

-- User-based collaborative filtering
SELECT 
    u1.user_name AS target_user,
    p.product_name AS recommended_product,
    AVG(pur.rating) AS avg_rating
FROM users u1
CROSS JOIN products p
LEFT JOIN purchases pur ON p.product_id = pur.product_id
WHERE p.product_id NOT IN (
    SELECT product_id FROM purchases WHERE user_id = u1.user_id
)
GROUP BY u1.user_name, p.product_name
HAVING AVG(pur.rating) > 3
ORDER BY u1.user_name, avg_rating DESC;

-- Customer segmentation using SQL (basic clustering)
SELECT 
    user_id,
    COUNT(*) AS total_purchases,
    AVG(rating) AS avg_rating,
    CASE 
        WHEN COUNT(*) > 3 AND AVG(rating) >= 4 THEN 'VIP Customer'
        WHEN COUNT(*) BETWEEN 2 AND 3 THEN 'Regular Customer' 
        ELSE 'New Customer'
    END AS customer_segment
FROM purchases 
GROUP BY user_id;