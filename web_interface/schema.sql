CREATE TABLE IF NOT EXISTS user_table(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    face TEXT NOT NULL, 
    sleep_or_read TEXT NOT NULL,
    ambient_noise TEXT NOT NULL
);

SELECT * FROM user_table;