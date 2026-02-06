import sqlite3

con = sqlite3.connect("senroll.db")
print("database opened successfully")

con.execute("create table ens (id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT NOT NULL, email TEXT UNIQUE NOT NULL ,address TEXT NOT NULL , number TEXT UNIQUE NOT NULL ,college_name TEXT NOT NULL ,city TEXT NOT NULL ,state TEXT NOT NULL)")

print("table created successsfully")

con.close()