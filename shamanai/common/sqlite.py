#!/usr/bin/env python3

'''
Thanks to Andres Torres
Source: https://www.pythoncentral.io/introduction-to-sqlite-in-python/
'''

import sqlite3

# Create a database in RAM
# db = sqlite3.connect(':memory:')

# Creates or opens a file called mydb with a SQLite3 DB
db = sqlite3.connect('db.sqlite3')

##########
# CREATE #
##########
cursor = db.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY,
        name TEXT,
        phone TEXT,
        email TEXT,
        password TEXT
    )
''')
db.commit()


##########
# INSERT #
##########
'''
If you need values from Python variables it is recommended to use the "?" placeholder.
Never use string operations or concatenation to make your queries because is very insecure. 
'''
cursor = db.cursor()
name = 'Andres'
phone = '3366858'
email = 'user@example.com'
password = '12345'
cursor.execute('''INSERT INTO users(name, phone, email, password)
                  VALUES(?,?,?,?)''', (name,phone, email, password))
db.commit()
'''
The values of the Python variables are passed inside a tuple.
Another way to do this is passing a dictionary using the ":keyname" placeholder:
'''
cursor = db.cursor()
cursor.execute('''INSERT INTO users(name, phone, email, password)
                  VALUES(:name,:phone, :email, :password)''',
                  {'name':name, 'phone':phone, 'email':email, 'password':password})
db.commit()

# If you need to insert several users use executemany and a list with the tuples:
users = [('a','1', 'a@b.com', 'a1'),
         ('b','2', 'b@b.com', 'b1'),
         ('c','3', 'c@b.com', 'c1'),
         ('c','3', 'c@b.com', 'c1')]
cursor.executemany(''' INSERT INTO users(name, phone, email, password) VALUES(?,?,?,?)''', users)
db.commit()


# ????
# If you need to get the id of the row you just inserted use lastrowid:
id = cursor.lastrowid
print('Last row id: %d' % id)


##########
# SELECT #
##########
# To retrieve data, execute the query against the cursor object
# and then use fetchone() to retrieve a single row or fetchall() to retrieve all the rows.

cursor.execute('''SELECT name, email, phone FROM users''')
user1 = cursor.fetchone() #retrieve the first row
print(user1[0])
all_rows = cursor.fetchall()
for row in all_rows:
    # row[0] returns the first column in the query (name), row[1] returns email column.
    print('{0} : {1}, {2}'.format(row[0], row[1], row[2]))
# The cursor object works as an iterator, invoking fetchall() automatically:
cursor.execute('''SELECT name, email, phone FROM users''')
for row in cursor:
    print('{0} : {1}, {2}'.format(row[0], row[1], row[2]))

# To retrive data with conditions, use again the "?" placeholder:
user_id = 3
cursor.execute('''SELECT name, email, phone FROM users WHERE id=?''', (user_id,))
user = cursor.fetchone()
db.commit()

##########
# UPDATE #
##########
# The procedure to update data is the same as inserting data:
newphone = '3113093164'
userid = 1
cursor.execute('''UPDATE users SET phone = ? WHERE id = ? ''', (newphone, userid))
db.commit()

##########
# DELETE #
##########
# The procedure to delete data is the same as inserting data:
delete_userid = 2
cursor.execute('''DELETE FROM users WHERE id = ? ''', (delete_userid,))
db.commit()





### About commit() and rollback():
'''
Using SQLite Transactions:
Transactions are an useful property of database systems.
It ensures the atomicity of the Database.
Use commit to save the changes.
Or rollback to roll back any change to the database since the last call to commit:
'''
cursor.execute('''UPDATE users SET phone = ? WHERE id = ? ''', (newphone, userid))
# The user's phone is not updated
db.rollback()

'''
Please remember to always call commit to save the changes.
If you close the connection using close or the connection to the file is lost
(maybe the program finishes unexpectedly), not committed changes will be lost.
'''



### Exception Handling:
try:
    db = sqlite3.connect('db.sqlite3')
    cursor = db.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS
                      users(id INTEGER PRIMARY KEY, name TEXT, phone TEXT, email TEXT, password TEXT)''')
    db.commit()
except Exception as e:
    # This is called a catch-all clause.
    # This is used here only as an example.
    # In a real application you should catch a specific exception such as IntegrityError or DatabaseError

    # Roll back any change if something goes wrong
    db.rollback()
    raise e
finally:
    db.close()

### SQLite Row Factory and Data Types
'''
The following table shows the relation between SQLite datatypes and Python datatypes:
    None type is converted to NULL
    int type is converted to INTEGER
    float type is converted to REAL
    str type is converted to TEXT
    bytes type is converted to BLOB
'''

# The row factory class sqlite3.Row is used to access the columns of a query by name instead of by index:
db = sqlite3.connect('db.sqlite3')
db.row_factory = sqlite3.Row
cursor = db.cursor()
cursor.execute('''SELECT name, email, phone FROM users''')
for row in cursor:
    # row['name'] returns the name column in the query, row['email'] returns email column.
    print('{0} -> {1}, {2}'.format(row['name'], row['email'], row['phone']))
db.close()



########
# DROP #
########
db = sqlite3.connect('db.sqlite3')
cursor = db.cursor()
cursor.execute('''DROP TABLE users''')
db.commit()

# When we are done working with the DB we need to close the connection:
db.close()