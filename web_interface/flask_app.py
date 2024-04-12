from flask import Flask, render_template, request, redirect, url_for
import sqlite3

app = Flask(__name__)

# Function to create a connection to the SQLite database
def get_db_connection():
    conn = sqlite3.connect('user_table.db')
    conn.row_factory = sqlite3.Row
    return conn

# Function to initialize the database
def init_db():
    conn = get_db_connection()
    with app.open_resource('schema.sql', mode='r') as f:
        conn.cursor().executescript(f.read())
    conn.commit()
    conn.close()

# Initialize the database
init_db()

# Route for the sign-up page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        face = request.files['face']
        sleep_or_read_setting = request.form['sleep_or_read']
        ambient_noise = request.form['ambient_noise']

        # Save the uploaded file
        face.save('uploads/' + face.filename)

        # Save the data to the database
        conn = get_db_connection()
        conn.execute('INSERT INTO user_table (name, face, sleep_or_read, ambient_noise) VALUES (?, ?, ?, ?)',
                     (name, face.filename, sleep_or_read_setting, ambient_noise))
        conn.commit()
        conn.close()

        return redirect(url_for('success'))

    return render_template('user_sign_up.html')

# Route for the success page
@app.route('/success')
def success():
    return 'Sign-up successful!'

if __name__ == '__main__':
    app.run(debug=True)