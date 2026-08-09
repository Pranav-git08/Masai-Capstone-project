import os
import sqlite3
import pandas as pd

GBP_TO_INR = 105.50

def main():
    data = {
        'Fiction': [
            ("A Light in the Attic", 51.77, "In stock", 3),
            ("Tipping the Velvet", 53.74, "In stock", 1),
            ("Soumission", 50.10, "In stock", 1),
            ("The Coming Woman", 17.93, "In stock", 3),
            ("The Boys in the Boat", 22.60, "In stock", 4),
            ("Sapiens", 54.23, "In stock", 5),
            ("Starving Hearts", 13.99, "In stock", 2),
            ("Shakespeare's Sonnets", 20.66, "In stock", 4),
            ("Set Me Free", 17.46, "In stock", 5),
            ("Scott Pilgrim's Precious Little Life", 52.29, "In stock", 5),
            ("Rip it Up and Start Again", 35.02, "In stock", 5),
            ("Our Band Could Be Your Life", 57.25, "In stock", 3),
            ("Olio", 23.88, "In stock", 1),
            ("Mesa Selimovic", 37.59, "In stock", 1),
            ("Libertarianism for Beginners", 51.33, "In stock", 2),
            ("It's Only Part of the Story", 45.00, "In stock", 5),
            ("How Music Works", 37.32, "In stock", 2),
            ("Foolproof Preserving", 30.52, "In stock", 3),
            ("Chase Me", 25.27, "In stock", 5),
            ("Black Dust", 34.53, "In stock", 5)
        ],
        'Science': [
            ("The Darkest Corners", 15.81, "In stock", 5),
            ("The Art of Field Coaching", 22.78, "In stock", 1),
            ("The Atom Bomb", 20.30, "Out of stock", 2),
            ("The Gene: An Intimate History", 25.99, "In stock", 5),
            ("Cosmos", 18.20, "In stock", 5),
            ("Astrophysics for People in a Hurry", 14.95, "In stock", 4),
            ("Brief Answers to the Big Questions", 16.50, "In stock", 4),
            ("The Selfish Gene", 19.99, "In stock", 3),
            ("What If?", 24.00, "In stock", 5),
            ("A Short History of Nearly Everything", 21.50, "In stock", 5),
            ("Sapiens: A Brief History", 22.00, "In stock", 4),
            ("The Elegant Universe", 17.80, "In stock", 3),
            ("Thinking, Fast and Slow", 15.20, "In stock", 4),
            ("Guns, Germs, and Steel", 18.90, "In stock", 4),
            ("The Order of Time", 12.99, "In stock", 5),
            ("Clean Code", 42.50, "In stock", 5),
            ("Design Patterns", 48.00, "In stock", 4),
            ("Structure and Interpretation of Computer Programs", 55.00, "In stock", 5),
            ("Introduction to Algorithms", 85.00, "In stock", 5),
            ("Deep Learning", 70.00, "In stock", 4)
        ],
        'History': [
            ("The Guns of August", 16.99, "In stock", 4),
            ("The Plantagenets", 18.50, "In stock", 3),
            ("SPQR: A History of Ancient Rome", 19.95, "In stock", 5),
            ("The Crusades", 22.40, "In stock", 4),
            ("1776", 17.50, "In stock", 4),
            ("Band of Brothers", 15.00, "In stock", 5),
            ("The Rise and Fall of the Third Reich", 28.00, "In stock", 4),
            ("Postwar", 25.00, "In stock", 5),
            ("The Civil War: A Narrative", 95.00, "In stock", 5),
            ("Stalin: Paradoxes of Power", 35.00, "In stock", 4),
            ("Churchill: A Life", 30.00, "In stock", 4),
            ("The Silk Roads", 18.00, "In stock", 5),
            ("Genoa and the Sea", 42.00, "In stock", 3),
            ("The Peloponnesian War", 21.00, "In stock", 4),
            ("A People's History of the United States", 20.00, "In stock", 5),
            ("The Twelve Caesars", 14.50, "In stock", 3),
            ("The Lessons of History", 12.00, "In stock", 5),
            ("Diplomacy", 26.00, "In stock", 4),
            ("The Cold War: A New History", 17.00, "In stock", 4),
            ("Catastrophe 1914", 22.50, "In stock", 4)
        ]
    }

    records = []
    cat_mapping = {}
    cat_id = 1

    for category_name, items in data.items():
        cat_mapping[category_name] = cat_id
        for title, price, stock, rating in items:
            records.append({
                'title': title,
                'category_id': cat_id,
                'price_gbp': float(price),
                'stock_status': stock,
                'rating': int(rating)
            })
        cat_id += 1

    df = pd.DataFrame(records)
    df['in_stock'] = df['stock_status'].str.strip().str.lower() == 'in stock'
    df['price_inr'] = (df['price_gbp'] * GBP_TO_INR).round(2)

    categories_df = pd.DataFrame([
        {'category_id': cid, 'category_name': name}
        for name, cid in cat_mapping.items()
    ])
    
    books_df = df[['title', 'category_id', 'price_gbp', 'price_inr', 'rating', 'in_stock']]

    os.makedirs('data_pipeline', exist_ok=True)
    db_file = 'data_pipeline/books_database.db'

    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS books;")
    cur.execute("DROP TABLE IF EXISTS categories;")

    cur.execute("""
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        );
    """)

    cur.execute("""
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            category_id INTEGER NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock BOOLEAN NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories (category_id)
        );
    """)

    categories_df.to_sql('categories', conn, if_exists='append', index=False)
    books_df.to_sql('books', conn, if_exists='append', index=False)
    conn.commit()

    q1 = "SELECT title, price_inr, rating FROM books WHERE rating = 5 ORDER BY price_inr DESC LIMIT 3;"
    q2 = "SELECT category_id, COUNT(*) as total_books, AVG(price_gbp) as avg_gbp FROM books GROUP BY category_id;"
    q3 = "SELECT category_id, AVG(rating) as avg_rating FROM books GROUP BY category_id HAVING avg_rating > 3.5;"
    q4 = "SELECT title, price_inr FROM books WHERE price_inr > (SELECT AVG(price_inr) FROM books) LIMIT 3;"
    
    q5 = """
        SELECT 
            b.book_id,
            b.title,
            c.category_name,
            b.price_gbp,
            b.price_inr,
            b.rating,
            b.in_stock
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY b.book_id ASC;
    """

    print("Query 1 Result:")
    print(pd.read_sql(q1, conn))
    print("\nQuery 2 Result:")
    print(pd.read_sql(q2, conn))
    print("\nQuery 3 Result:")
    print(pd.read_sql(q3, conn))
    print("\nQuery 4 Result:")
    print(pd.read_sql(q4, conn))
    
    print("\nQuery 5 (SQL Join) Result:")
    sql_joined = pd.read_sql(q5, conn)
    print(sql_joined.head())

    books_sql = pd.read_sql("SELECT * FROM books;", conn)
    cats_sql = pd.read_sql("SELECT * FROM categories;", conn)

    pandas_merged = pd.merge(
        books_sql, 
        cats_sql, 
        on='category_id', 
        how='inner'
    )[['book_id', 'title', 'category_name', 'price_gbp', 'price_inr', 'rating', 'in_stock']].sort_values('book_id').reset_index(drop=True)

    sql_joined['in_stock'] = sql_joined['in_stock'].astype(bool)
    pandas_merged['in_stock'] = pandas_merged['in_stock'].astype(bool)

    print("\nEquivalency Check:")
    print(sql_joined.equals(pandas_merged))

    conn.close()

if __name__ == '__main__':
    main() 