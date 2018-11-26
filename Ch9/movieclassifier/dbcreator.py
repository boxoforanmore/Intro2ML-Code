# Creates a new sqlite3 db

import sqlite3
import os

if os.path.exists('reviews.sqlite'):
    os.remove('reviews.sqlite')

conn = sqlite3.connect('reviews.sqlite')
c = conn.cursor()
c.execute('CREATE TABLE review_db'\
          ' (review TEXT, sentiment INTEGER, date TEXT)')

# Inserts example data
example1 = 'I love this movie'
example2 = 'I disliked this movie'

c.execute("INSERT INTO review_db" \
          " (review, sentiment, date) VALUES" \
          " (?, ?, DATETIME('now'))", (example1, 1))

c.execute("INSERT INTO review_db" \
          " (review, sentiment, date) VALUES" \
          " (?, ?, DATETIME('now'))", (example2, 0))

conn.commit()
conn.close()


# Check that the entries have been stored correctly
conn = sqlite3.connect('reviews.sqlite')
c = conn.cursor()
c.execute("SELECT * FROM review_db WHERE date" \
          " BETWEEN '2017-01-01 00:00:00' AND DATETIME('now')")

results = c.fetchall()
conn.close()
print(f"Results of SQL call: \n {results}")
