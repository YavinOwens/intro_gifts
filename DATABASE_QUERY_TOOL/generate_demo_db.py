import sqlite3
import random
import string
import datetime
import os
import pandas as pd
from faker import Faker

DB_PATH = os.path.join(os.path.dirname(__file__), 'road_network_demo.db')
fake = Faker()

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Drop tables if they exist
cursor.execute('DROP TABLE IF EXISTS roads')
cursor.execute('DROP TABLE IF EXISTS maintenance_events')
cursor.execute('DROP TABLE IF EXISTS contractors')
cursor.execute('DROP TABLE IF EXISTS inspections')

# Create tables
cursor.execute('''
CREATE TABLE roads (
    road_id INTEGER PRIMARY KEY,
    name TEXT,
    length_km REAL,
    lanes INTEGER,
    surface_type TEXT,
    city TEXT
)
''')

cursor.execute('''
CREATE TABLE contractors (
    contractor_id INTEGER PRIMARY KEY,
    name TEXT,
    contact_email TEXT,
    rating REAL
)
''')

cursor.execute('''
CREATE TABLE maintenance_events (
    event_id INTEGER PRIMARY KEY,
    road_id INTEGER,
    contractor_id INTEGER,
    event_type TEXT,
    start_date TEXT,
    end_date TEXT,
    cost REAL,
    FOREIGN KEY(road_id) REFERENCES roads(road_id),
    FOREIGN KEY(contractor_id) REFERENCES contractors(contractor_id)
)
''')

cursor.execute('''
CREATE TABLE inspections (
    inspection_id INTEGER PRIMARY KEY,
    road_id INTEGER,
    inspection_date TEXT,
    inspector TEXT,
    issues_found INTEGER,
    FOREIGN KEY(road_id) REFERENCES roads(road_id)
)
''')

surfaces = ['asphalt', 'concrete', 'gravel', 'cobblestone']
event_types = ['resurfacing', 'pothole repair', 'line painting', 'bridge repair', 'signage update']

# Insert roads
for i in range(1, 1001):
    cursor.execute('INSERT INTO roads (road_id, name, length_km, lanes, surface_type, city) VALUES (?, ?, ?, ?, ?, ?)',
                   (i, f"Road-{fake.bothify(text='??###')}", round(random.uniform(0.5, 25.0), 2), random.randint(1, 6), random.choice(surfaces), fake.city()))

# Insert contractors
for i in range(1, 1001):
    name = fake.company()
    email = fake.company_email()
    rating = round(random.uniform(2.5, 5.0), 2)
    cursor.execute('INSERT INTO contractors (contractor_id, name, contact_email, rating) VALUES (?, ?, ?, ?)',
                   (i, name, email, rating))

# Insert maintenance_events
for i in range(1, 1001):
    road_id = random.randint(1, 1000)
    contractor_id = random.randint(1, 1000)
    event_type = random.choice(event_types)
    start = datetime.date(2020, 1, 1) + datetime.timedelta(days=random.randint(0, 1460))
    end = start + datetime.timedelta(days=random.randint(1, 30))
    cost = round(random.uniform(1000, 50000), 2)
    cursor.execute('INSERT INTO maintenance_events (event_id, road_id, contractor_id, event_type, start_date, end_date, cost) VALUES (?, ?, ?, ?, ?, ?, ?)',
                   (i, road_id, contractor_id, event_type, str(start), str(end), cost))

# Insert inspections
for i in range(1, 1001):
    road_id = random.randint(1, 1000)
    date = datetime.date(2020, 1, 1) + datetime.timedelta(days=random.randint(0, 1460))
    inspector = fake.name()
    issues = random.randint(0, 10)
    cursor.execute('INSERT INTO inspections (inspection_id, road_id, inspection_date, inspector, issues_found) VALUES (?, ?, ?, ?, ?)',
                   (i, road_id, str(date), inspector, issues))

conn.commit()
conn.close()

print(f"Demo database created at {DB_PATH}") 