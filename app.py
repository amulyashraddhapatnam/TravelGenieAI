import os
import json
import sqlite3
import requests
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, g
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "travelgenie_secret_amulya_2026")
DATABASE = 'travelgenie.db'

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# --- SQLite Connection Helpers ---
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                destination TEXT NOT NULL,
                days INTEGER NOT NULL,
                start_date TEXT DEFAULT '',
                end_date TEXT DEFAULT '',
                num_travelers INTEGER DEFAULT 1,
                budget_amount REAL NOT NULL,
                currency TEXT NOT NULL DEFAULT 'INR',
                travel_type TEXT NOT NULL,
                interests TEXT NOT NULL,
                dietary_prefs TEXT DEFAULT '',
                accessibility_prefs TEXT DEFAULT '',
                is_student INTEGER DEFAULT 0,
                is_favorite INTEGER DEFAULT 0,
                notes TEXT DEFAULT '',
                checklist TEXT DEFAULT '[]',
                itinerary_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')
        db.commit()

init_db()

# --- Helpers ---
def get_weather(destination):
    if not WEATHER_API_KEY:
        return {
            "current_temp": "28 °C",
            "forecast": "Sunny / Clear Sky",
            "humidity": "58%",
            "wind_speed": "12 km/h",
            "rain_chance": "10%",
            "best_time": "October to March"
        }
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={destination}&appid={WEATHER_API_KEY}&units=metric"
        res = requests.get(url, timeout=5).json()
        if res.get("cod") == 200:
            return {
                "current_temp": f"{round(res['main']['temp'])} °C",
                "forecast": res['weather'][0]['description'].capitalize(),
                "humidity": f"{res['main']['humidity']}%",
                "wind_speed": f"{round(res['wind']['speed'] * 3.6)} km/h",
                "rain_chance": f"{res.get('clouds', {}).get('all', 15)}%",
                "best_time": "Optimal Season"
            }
    except Exception:
        pass
    return {
        "current_temp": "26 °C",
        "forecast": "Pleasant",
        "humidity": "60%",
        "wind_speed": "10 km/h",
        "rain_chance": "15%",
        "best_time": "October to March"
    }

def generate_ai_itinerary(destination, days, start_date, end_date, num_travelers, budget_amount, currency, travel_type, interests, dietary, accessibility, is_student=False, custom_prompt=""):
    per_person_budget = round(budget_amount / num_travelers, 2)
    
    student_instruction = ""
    if is_student:
        student_instruction = """
        [CRITICAL STUDENT MODE INSTRUCTIONS]: 
        The travelers are students carrying valid College IDs. You MUST explicitly mention student ID-based discounts wherever applicable in the itinerary.
        - Focus on student-friendly options (hostels, dorms, street food, group splits, public buses/metros, student discounts).
        - Highlight attractions, amusement parks, museums, or historical monuments that offer student price cuts/concessions with a valid College ID.
        - Recommend student-friendly transit passes (like youth rail passes or metro student concessions).
        - Factor these lower student-ticket prices directly into the estimated daily costs.
        """

    prompt = f"""
You are an expert travel consultant for TravelGenie AI. Create a travel itinerary in JSON for a trip to {destination} for {days} days.
Travel Dates: From {start_date} to {end_date}.
Number of Travelers: {num_travelers} person(s).
Total Manual Budget: {currency} {budget_amount} (which is roughly {currency} {per_person_budget} per person).
Travel Style: {travel_type}.
Interests: {interests}.
Dietary Preferences: {dietary}.
Accessibility/Group Needs: {accessibility}.
{student_instruction}
{custom_prompt}

Return ONLY valid JSON matching this schema exactly (no markdown fences, pure JSON):
{{
  "destination": "{destination}",
  "days": {days},
  "start_date": "{start_date}",
  "end_date": "{end_date}",
  "num_travelers": {num_travelers},
  "currency": "{currency}",
  "total_budget": {budget_amount},
  "per_person_budget": {per_person_budget},
  "estimated_total_cost": "{currency} {budget_amount}",
  "currency_info": {{
    "currency_name": "{currency}",
    "exchange_rate": "1 USD ≈ ₹83.5 INR"
  }},
  "detailed_weather": {{
    "current_temp": "28 °C",
    "forecast": "Sunny",
    "humidity": "60%",
    "wind_speed": "12 km/h",
    "rain_chance": "10%",
    "best_time_to_travel": "October to March"
  }},
  "budget_breakdown": {{
    "hotel": {round(budget_amount * 0.4)},
    "food": {round(budget_amount * 0.25)},
    "transport": {round(budget_amount * 0.15)},
    "tickets": {round(budget_amount * 0.1)},
    "shopping": {round(budget_amount * 0.05)},
    "emergency": {round(budget_amount * 0.05)},
    "total": {budget_amount}
  }},
  "latitude": 20.5937,
  "longitude": 78.9629,
  "safety_and_scam_warnings": [
    "Keep personal belongings secure in crowded markets",
    "Use officially registered cabs or metro"
  ],
  "transit_guide": {{
    "flight_cost": "{currency} {round(budget_amount * 0.2)}",
    "train_cost": "{currency} {round(budget_amount * 0.08)}",
    "bus_cost": "{currency} {round(budget_amount * 0.04)}",
    "nearest_airport": "Main International/Domestic Airport",
    "nearest_railway": "Central Railway Station",
    "local_transport": "Metro, Cabs, Auto-rickshaws"
  }},
  "ai_suggestions": {{
    "best_month": "October to March",
    "local_dish_must_try": "Popular Regional Delicacy",
    "pro_tip": "Book entry tickets online in advance to skip lines."
  }},
  "hotel_recommendations": [
    {{
      "name": "Grand City Hotel",
      "rating": "4.5 ⭐",
      "price": "{currency} {round(budget_amount * 0.1)}/night",
      "distance": "1.2 km from center",
      "amenities": "Free WiFi, Breakfast, AC"
    }}
  ],
  "restaurant_suggestions": {{
    "breakfast": [{{"name": "Morning Cafe", "dish": "Fresh Breakfast Set", "cost": "{currency} 300", "rating": "4.6 ⭐"}}],
    "lunch": [{{"name": "Bistro Delight", "dish": "Local Special Thali/Platter", "cost": "{currency} 600", "rating": "4.5 ⭐"}}],
    "dinner": [{{"name": "Sunset Dining", "dish": "Chef Special", "cost": "{currency} 900", "rating": "4.7 ⭐"}}],
    "famous_local_dishes": ["Signature Street Snack", "Traditional Dessert"],
    "budget_restaurants": [{{"name": "Corner Eatery", "cost": "{currency} 200", "rating": "4.3 ⭐"}}],
    "premium_restaurants": [{{"name": "Royal Fine Dining", "cost": "{currency} 2200", "rating": "4.8 ⭐"}}]
  }},
  "map_pins": [
    {{"name": "Central City Square", "lat": 20.59, "lng": 78.96, "category": "Spot"}}
  ],
  "day_by_day": [
    {{
      "day": 1,
      "date": "{start_date}",
      "theme": "Arrival & City Highlights",
      "morning": "Morning exploration at main city landmark",
      "lunch": "Lunch at highly-rated local bistro",
      "afternoon": "Visit prominent cultural spot/museum",
      "evening": "Sunset walk and local market shopping",
      "estimated_cost": "{currency} {round(budget_amount / days)}"
    }}
  ],
  "packing_checklist": [
    {{"item": "Comfortable Walking Shoes", "category": "Essential", "checked": false}}
  ]
}}
"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    res_data = response.json()
    
    try:
        clean_text = res_data['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception:
        clean_text = "{}"

    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    if clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    
    clean_text = clean_text.strip()
    
    try:
        json.loads(clean_text)
        return clean_text
    except json.JSONDecodeError:
        return json.dumps({"itinerary": clean_text})

# --- Routes ---
@app.route('/')
def home():
    popular_destinations = [
        {"name": "Goa", "country": "India", "rating": "4.8", "best_time": "Nov - Feb", "budget": "₹12,000", "image": "[https://images.unsplash.com/photo-1512343879784-a960bf40e7f2?w=500&auto=format&fit=crop&q=60](https://images.unsplash.com/photo-1512343879784-a960bf40e7f2?w=500&auto=format&fit=crop&q=60)"},
        {"name": "Jaipur", "country": "India", "rating": "4.7", "best_time": "Oct - Mar", "budget": "₹10,000", "image": "[https://images.unsplash.com/photo-1477587458883-47145ed94245?w=500&auto=format&fit=crop&q=60](https://images.unsplash.com/photo-1477587458883-47145ed94245?w=500&auto=format&fit=crop&q=60)"},
        {"name": "Manali", "country": "India", "rating": "4.9", "best_time": "Oct - Jun", "budget": "₹14,000", "image": "[https://images.unsplash.com/photo-1626621341517-bbf3d9990a23?w=500&auto=format&fit=crop&q=60](https://images.unsplash.com/photo-1626621341517-bbf3d9990a23?w=500&auto=format&fit=crop&q=60)"},
        {"name": "Kerala", "country": "India", "rating": "4.8", "best_time": "Sep - Mar", "budget": "₹15,000", "image": "[https://images.unsplash.com/photo-1602216056096-3b40cc0c9944?w=500&auto=format&fit=crop&q=60](https://images.unsplash.com/photo-1602216056096-3b40cc0c9944?w=500&auto=format&fit=crop&q=60)"}
    ]
    return render_template('index.html', popular=popular_destinations)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip()
        password = request.form['password']
        
        db = get_db()
        cursor = db.cursor()
        if cursor.execute('SELECT id FROM users WHERE username = ? OR email = ?', (username, email)).fetchone():
            flash('Username or Email already exists.', 'danger')
            return redirect(url_for('register'))
            
        hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
        cursor.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)', (username, email, hashed_pw))
        db.commit()
        flash('Account created! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        
        db = get_db()
        cursor = db.cursor()
        
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        row = cursor.fetchone()
        
        if row:
            user = dict(row) if hasattr(row, 'keys') else {
                'id': row[0], 
                'username': row[1], 
                'email': row[2], 
                'password_hash': row[3]
            }
            
            if check_password_hash(user['password_hash'], password):
                session['user_id'] = user['id']
                session['username'] = user['username']  
                flash('Successfully logged in!', 'success')
                return redirect(url_for('planner'))
                
        flash('Invalid email or password.', 'danger')
    return render_template('login.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        flash('If an account with that email exists, password reset instructions have been sent.', 'info')
        return redirect(url_for('login'))
    return render_template('forgot_password.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('home'))

@app.route('/planner', methods=['GET', 'POST'])
def planner():
    if 'user_id' not in session:
        flash('Please login to plan your trip.', 'warning')
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        destination = request.form.get('destination', '').strip()
        days = int(request.form.get('days', 3))
        start_date_str = request.form.get('start_date', datetime.today().strftime('%Y-%m-%d'))
        num_travelers = int(request.form.get('num_travelers', 1))
        budget_amount = float(request.form.get('budget_amount', 15000))
        currency = request.form.get('currency', 'INR')
        travel_type = request.form.get('travel_type', 'Friends Group')
        interests = ", ".join(request.form.getlist('interests')) or "Sightseeing"
        dietary = ", ".join(request.form.getlist('dietary')) or "Flexible"
        accessibility = ", ".join(request.form.getlist('accessibility')) or "None"
        is_student = 1 if 'is_student' in request.form else 0

        # --- AI-POWERED BUDGET FEASIBILITY CHECK ---
        try:
            check_url = f"[https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=](https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=){GEMINI_API_KEY}"
            check_prompt = f"Act as a strict financial travel auditor. Can a trip for {num_travelers} traveler(s) to {destination} for {days} days be reasonably done on a total budget of {currency} {budget_amount}? Answer ONLY 'YES' or 'NO'."
            check_res = requests.post(check_url, headers={"Content-Type": "application/json"}, json={"contents": [{"parts": [{"text": check_prompt}]}]}, timeout=10).json()
            
            check_text = "YES"
            try:
                check_text = check_res['candidates'][0]['content']['parts'][0]['text'].strip().upper()
            except (KeyError, IndexError, TypeError):
                check_text = "YES"
            
            if "NO" in check_text:
                flash(f"⚠️ AI Budget Check Failed: {currency} {budget_amount} is severely underfunded for a {days}-day trip to {destination} for {num_travelers} traveler(s). Please increase your budget!", "danger")
                return redirect(url_for('planner'))
        except Exception as e:
            print("Budget check error:", e)

        # Calculate End Date automatically
        try:
            start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date_obj = start_date_obj + timedelta(days=days - 1)
            end_date_str = end_date_obj.strftime('%b %d, %Y')
            formatted_start_date = start_date_obj.strftime('%b %d, %Y')
        except Exception:
            formatted_start_date = start_date_str
            end_date_str = "Calculated End Date"

        try:
            raw_json = generate_ai_itinerary(
                destination, days, formatted_start_date, end_date_str, num_travelers, 
                budget_amount, currency, travel_type, interests, dietary, accessibility, bool(is_student)
            )
            parsed_data = json.loads(raw_json)
            checklist_json = json.dumps(parsed_data.get('packing_checklist', []))
            
            db = get_db()
            cursor = db.cursor()
            cursor.execute('''
                INSERT INTO trips (user_id, destination, days, start_date, end_date, num_travelers, budget_amount, currency, travel_type, interests, dietary_prefs, accessibility_prefs, is_student, checklist, itinerary_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (session['user_id'], destination, days, formatted_start_date, end_date_str, num_travelers, budget_amount, currency, travel_type, interests, dietary, accessibility, is_student, checklist_json, raw_json))
            db.commit()
            
            return redirect(url_for('itinerary', trip_id=cursor.lastrowid))
        except Exception as e:
            flash(f'Error generating itinerary: {str(e)}', 'danger')
            return redirect(url_for('planner'))

    preset_destination = request.args.get('destination', '')
    today_date = datetime.today().strftime('%Y-%m-%d')
    return render_template('planner.html', preset_destination=preset_destination, today_date=today_date)

@app.route('/itinerary/<int:trip_id>')
def itinerary(trip_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    db = get_db()
    trip = db.cursor().execute('SELECT * FROM trips WHERE id = ? AND user_id = ?', (trip_id, session['user_id'])).fetchone()
    if not trip:
        flash('Trip not found.', 'danger')
        return redirect(url_for('my_trips'))

    try:
        itinerary_data = json.loads(trip['itinerary_json'])
    except Exception:
        itinerary_data = {}

    # --- SAFETY FALLBACKS FOR MISSING SECTIONS ---
    itinerary_data.setdefault('hotel_recommendations', [
        {"name": "Recommended Hotel", "rating": "4.5 ⭐", "price": "Standard", "distance": "Central", "amenities": "WiFi, AC"}
    ])
    itinerary_data.setdefault('restaurant_suggestions', {
        "breakfast": [{"name": "Local Cafe", "dish": "Breakfast Combo", "cost": "Moderate", "rating": "4.5 ⭐"}],
        "lunch": [{"name": "City Bistro", "dish": "Special Thali", "cost": "Moderate", "rating": "4.5 ⭐"}],
        "dinner": [{"name": "Sunset Diner", "dish": "Chef Special", "cost": "Moderate", "rating": "4.6 ⭐"}],
        "famous_local_dishes": ["Regional Specialty"],
        "budget_restaurants": [],
        "premium_restaurants": []
    })
    itinerary_data.setdefault('packing_checklist', [
        {"item": "Comfortable Walking Shoes", "category": "Essential", "checked": False}
    ])

    weather_data = get_weather(trip['destination'])
    return render_template('itinerary.html', trip=trip, data=itinerary_data, weather=weather_data)

@app.route('/api/trip/<int:trip_id>/notes', methods=['POST'])
def update_notes(trip_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    req_data = request.get_json()
    notes = req_data.get('notes', '')
    
    db = get_db()
    db.cursor().execute('UPDATE trips SET notes = ? WHERE id = ? AND user_id = ?', (notes, trip_id, session['user_id']))
    db.commit()
    return jsonify({'success': True})

@app.route('/api/trip/<int:trip_id>/favorite', methods=['POST'])
def toggle_favorite(trip_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
        
    db = get_db()
    cursor = db.cursor()
    trip = cursor.execute('SELECT is_favorite FROM trips WHERE id = ? AND user_id = ?', (trip_id, session['user_id'])).fetchone()
    
    if not trip:
        return jsonify({'success': False, 'error': 'Not found'}), 404

    new_fav = 0 if trip['is_favorite'] else 1
    cursor.execute('UPDATE trips SET is_favorite = ? WHERE id = ? AND user_id = ?', (new_fav, trip_id, session['user_id']))
    db.commit()
    return jsonify({'success': True, 'is_favorite': bool(new_fav)})

@app.route('/api/trip/<int:trip_id>/optimize', methods=['POST'])
def optimize_trip(trip_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
        
    db = get_db()
    trip = db.cursor().execute('SELECT * FROM trips WHERE id = ? AND user_id = ?', (trip_id, session['user_id'])).fetchone()
    if not trip:
        return jsonify({'success': False, 'error': 'Trip not found'}), 404

    prompt_modifier = request.json.get('prompt', '')
    
    try:
        raw_json = generate_ai_itinerary(
            trip['destination'], trip['days'], trip['start_date'], trip['end_date'], trip['num_travelers'],
            trip['budget_amount'], trip['currency'], trip['travel_type'],
            trip['interests'], trip['dietary_prefs'], trip['accessibility_prefs'], bool(trip['is_student']),
            custom_prompt=f"REFINEMENT: {prompt_modifier}"
        )
        db.cursor().execute('UPDATE trips SET itinerary_json = ? WHERE id = ? AND user_id = ?', (raw_json, trip_id, session['user_id']))
        db.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/my_trips')
def my_trips():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    db = get_db()
    user_trips = db.cursor().execute('SELECT * FROM trips WHERE user_id = ? ORDER BY created_at DESC', (session['user_id'],)).fetchall()
    
    total_trips = len(user_trips)
    favorites_count = sum(1 for t in user_trips if t['is_favorite'])
    total_budget_sum = sum(t['budget_amount'] for t in user_trips)
    
    return render_template('my_trips.html', trips=user_trips, total=total_trips, favorites=favorites_count, total_budget=total_budget_sum)

@app.route('/delete_trip/<int:trip_id>', methods=['POST'])
def delete_trip(trip_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    db = get_db()
    db.cursor().execute('DELETE FROM trips WHERE id = ? AND user_id = ?', (trip_id, session['user_id']))
    db.commit()
    flash('Trip deleted successfully.', 'info')
    return redirect(url_for('my_trips'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
