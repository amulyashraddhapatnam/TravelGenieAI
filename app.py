import os
import json
import time
import sqlite3
import requests
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, g
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "travelgenie_secret_amulya_2026")
DATABASE = os.getenv("DATABASE_PATH", "travelgenie.db")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"


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


# --- Weather helper ---
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
        res = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": destination, "appid": WEATHER_API_KEY, "units": "metric"},
            timeout=5
        ).json()
        if res.get("cod") == 200:
            return {
                "current_temp": f"{round(res['main']['temp'])} °C",
                "forecast": res['weather'][0]['description'].capitalize(),
                "humidity": f"{res['main']['humidity']}%",
                "wind_speed": f"{round(res['wind']['speed'] * 3.6)} km/h",
                "rain_chance": f"{res.get('clouds', {}).get('all', 15)}%",
                "best_time": "Optimal Season"
            }
    except Exception as e:
        print("Weather API error:", e)
    return {
        "current_temp": "26 °C",
        "forecast": "Pleasant",
        "humidity": "60%",
        "wind_speed": "10 km/h",
        "rain_chance": "15%",
        "best_time": "October to March"
    }


# --- Gemini helpers ---
def call_gemini(prompt, json_mode=True, timeout=60, retries=2):
    """Call Gemini and return the raw text. Raises RuntimeError if every attempt fails."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")

    generation_config = {
        "temperature": 0.7,
        "maxOutputTokens": 8192,
        # Disable "thinking" so reasoning tokens don't eat the output budget / add latency.
        "thinkingConfig": {"thinkingBudget": 0},
    }
    if json_mode:
        generation_config["responseMimeType"] = "application/json"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=timeout)
            res_data = response.json()

            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {res_data.get('error', res_data)}")

            candidates = res_data.get('candidates', [])
            if not candidates:
                raise RuntimeError(f"No candidates returned: {res_data.get('promptFeedback', res_data)}")

            parts = candidates[0].get('content', {}).get('parts', [])
            text = "".join(p.get('text', '') for p in parts).strip()
            if not text:
                raise RuntimeError(f"Empty response (finishReason={candidates[0].get('finishReason')})")
            return text
        except Exception as e:
            last_error = e
            print(f"Gemini call failed (attempt {attempt + 1}/{retries + 1}):", e)
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Gemini request failed: {last_error}")


def extract_json(text):
    """Parse JSON from model output, tolerating markdown fences and surrounding prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def build_itinerary_template(destination, days, start_date, end_date, num_travelers,
                             budget_amount, currency, day_count=None):
    """Complete itinerary structure. Used as the JSON schema for the prompt,
    as the fallback when the AI fails, and to fill any sections the AI leaves out."""
    per_person_budget = round(budget_amount / max(num_travelers, 1), 2)
    day_count = days if day_count is None else day_count

    try:
        start_obj = datetime.strptime(start_date, '%b %d, %Y')
    except Exception:
        start_obj = None

    day_by_day = []
    for i in range(day_count):
        date_label = (start_obj + timedelta(days=i)).strftime('%b %d, %Y') if start_obj else start_date
        day_by_day.append({
            "day": i + 1,
            "date": date_label,
            "theme": "Arrival & City Highlights" if i == 0 else f"Explore {destination} - Day {i + 1}",
            "morning": f"Morning exploration at a main landmark in {destination}",
            "lunch": "Lunch at a highly-rated local restaurant",
            "afternoon": "Visit a prominent cultural spot or museum",
            "evening": "Sunset walk and local market shopping",
            "estimated_cost": f"{currency} {round(budget_amount / max(days, 1))}"
        })

    return {
        "destination": destination,
        "days": days,
        "start_date": start_date,
        "end_date": end_date,
        "num_travelers": num_travelers,
        "currency": currency,
        "total_budget": budget_amount,
        "per_person_budget": per_person_budget,
        "estimated_total_cost": f"{currency} {budget_amount}",
        "currency_info": {
            "currency_name": currency,
            "exchange_rate": "1 USD ≈ ₹83.5 INR"
        },
        "detailed_weather": {
            "current_temp": "28 °C",
            "forecast": "Sunny",
            "humidity": "60%",
            "wind_speed": "12 km/h",
            "rain_chance": "10%",
            "best_time_to_travel": "October to March"
        },
        "budget_breakdown": {
            "hotel": round(budget_amount * 0.4),
            "food": round(budget_amount * 0.25),
            "transport": round(budget_amount * 0.15),
            "tickets": round(budget_amount * 0.1),
            "shopping": round(budget_amount * 0.05),
            "emergency": round(budget_amount * 0.05),
            "total": budget_amount
        },
        "latitude": 20.5937,
        "longitude": 78.9629,
        "safety_and_scam_warnings": [
            "Keep personal belongings secure in crowded markets",
            "Use officially registered cabs or metro"
        ],
        "transit_guide": {
            "flight_cost": f"{currency} {round(budget_amount * 0.2)}",
            "train_cost": f"{currency} {round(budget_amount * 0.08)}",
            "bus_cost": f"{currency} {round(budget_amount * 0.04)}",
            "nearest_airport": "Main International/Domestic Airport",
            "nearest_railway": "Central Railway Station",
            "local_transport": "Metro, Cabs, Auto-rickshaws"
        },
        "ai_suggestions": {
            "best_month": "October to March",
            "local_dish_must_try": "Popular Regional Delicacy",
            "pro_tip": "Book entry tickets online in advance to skip lines."
        },
        "hotel_recommendations": [{
            "name": "Grand City Hotel",
            "rating": "4.5 ⭐",
            "price": f"{currency} {round(budget_amount * 0.1)}/night",
            "distance": "1.2 km from center",
            "amenities": "Free WiFi, Breakfast, AC"
        }],
        "restaurant_suggestions": {
            "breakfast": [{"name": "Morning Cafe", "dish": "Fresh Breakfast Set", "cost": f"{currency} 300", "rating": "4.6 ⭐"}],
            "lunch": [{"name": "Bistro Delight", "dish": "Local Special Thali/Platter", "cost": f"{currency} 600", "rating": "4.5 ⭐"}],
            "dinner": [{"name": "Sunset Dining", "dish": "Chef Special", "cost": f"{currency} 900", "rating": "4.7 ⭐"}],
            "famous_local_dishes": ["Signature Street Snack", "Traditional Dessert"],
            "budget_restaurants": [{"name": "Corner Eatery", "cost": f"{currency} 200", "rating": "4.3 ⭐"}],
            "premium_restaurants": [{"name": "Royal Fine Dining", "cost": f"{currency} 2200", "rating": "4.8 ⭐"}]
        },
        "map_pins": [
            {"name": "Central City Square", "lat": 20.59, "lng": 78.96, "category": "Spot"}
        ],
        "day_by_day": day_by_day,
        "packing_checklist": [
            {"item": "Comfortable Walking Shoes", "category": "Essential", "checked": False},
            {"item": "Valid Government ID / Passport & College ID", "category": "Documents", "checked": False},
            {"item": "Smartphone & Portable Power Bank (10,000mAh)", "category": "Electronics", "checked": False},
            {"item": "Universal Charger & USB Cables", "category": "Electronics", "checked": False},
            {"item": "Prescription Medications & Basic First Aid Kit", "category": "Health", "checked": False},
            {"item": "Sunscreen (SPF 50+) & Sunglasses", "category": "Personal Care", "checked": False},
            {"item": "Reusable Water Bottle", "category": "Gear", "checked": False},
            {"item": "Compact Umbrella / Rain Poncho", "category": "Weather", "checked": False},
            {"item": "Travel-sized Toiletries & Hand Sanitizer", "category": "Personal Care", "checked": False},
            {"item": "Lightweight Daypack / Sling Bag", "category": "Gear", "checked": False}
        ]
    }


def normalize_itinerary(data, template):
    """Fill every missing / empty / wrongly-typed section of `data` from `template`
    so the itinerary page never renders blank sections."""
    if not isinstance(data, dict):
        return template

    for key, default in template.items():
        value = data.get(key)
        if value in (None, "", [], {}) or not isinstance(value, type(default)):
            # allow ints for float defaults and vice versa
            if isinstance(default, (int, float)) and isinstance(value, (int, float)) and not isinstance(value, bool):
                continue
            data[key] = default
        elif isinstance(default, dict):
            for sub_key, sub_default in default.items():
                if data[key].get(sub_key) in (None, "", [], {}):
                    data[key][sub_key] = sub_default

    # Make sure there is one entry per day.
    days = template.get("days", 0)
    day_list = data["day_by_day"]
    if len(day_list) < days:
        for filler in template["day_by_day"][len(day_list):days]:
            day_list.append(filler)
    return data


def generate_ai_itinerary(destination, days, start_date, end_date, num_travelers, budget_amount,
                          currency, travel_type, interests, dietary, accessibility,
                          is_student=False, custom_prompt="", allow_fallback=True):
    """Return a JSON string for the itinerary. Never returns an empty object.
    If the AI fails and allow_fallback is True, returns a complete default itinerary
    flagged with "_fallback": true. If allow_fallback is False, raises instead."""
    per_person_budget = round(budget_amount / max(num_travelers, 1), 2)
    full_template = build_itinerary_template(destination, days, start_date, end_date,
                                             num_travelers, budget_amount, currency)
    schema_example = build_itinerary_template(destination, days, start_date, end_date,
                                              num_travelers, budget_amount, currency, day_count=1)

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

Return ONLY valid JSON (no markdown fences, no commentary) with exactly the structure below.
Replace ALL placeholder values with real, specific information about {destination}: real hotels, restaurants,
attractions, accurate latitude/longitude for {destination}, realistic weather, and map_pins for at least 5 real places.
"day_by_day" MUST contain exactly {days} entries (day 1 through day {days}), each with a different theme and
specific places. The budget_breakdown values must add up to the total budget.

{json.dumps(schema_example, ensure_ascii=False, indent=2)}
"""

    try:
        raw_text = call_gemini(prompt, json_mode=True, timeout=60, retries=1)
        parsed = extract_json(raw_text)
        if not isinstance(parsed, dict) or not parsed:
            raise ValueError("Model returned an empty or non-object JSON response")
        parsed = normalize_itinerary(parsed, full_template)
        return json.dumps(parsed, ensure_ascii=False)
    except Exception as e:
        print("Itinerary generation failed:", e)
        if not allow_fallback:
            raise
        fallback = full_template
        fallback["_fallback"] = True
        return json.dumps(fallback, ensure_ascii=False)


# --- Routes ---
@app.route('/')
def home():
    popular_destinations = [
        {"name": "Goa", "country": "India", "rating": "4.8", "best_time": "Nov - Feb", "budget": "₹12,000", "image": "https://images.unsplash.com/photo-1512343879784-a960bf40e7f2?w=500&auto=format&fit=crop&q=60"},
        {"name": "Jaipur", "country": "India", "rating": "4.7", "best_time": "Oct - Mar", "budget": "₹10,000", "image": "https://images.unsplash.com/photo-1477587458883-47145ed94245?w=500&auto=format&fit=crop&q=60"},
        {"name": "Manali", "country": "India", "rating": "4.9", "best_time": "Oct - Jun", "budget": "₹14,000", "image": "https://images.unsplash.com/photo-1626621341517-bbf3d9990a23?w=500&auto=format&fit=crop&q=60"},
        {"name": "Kerala", "country": "India", "rating": "4.8", "best_time": "Sep - Mar", "budget": "₹15,000", "image": "https://images.unsplash.com/photo-1602216056096-3b40cc0c9944?w=500&auto=format&fit=crop&q=60"}
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
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        db = get_db()
        user = db.cursor().execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()

        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Successfully logged in!', 'success')
            return redirect(url_for('planner'))
        flash('Invalid email or password.', 'danger')
    return render_template('login.html')


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
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
        # --- Input parsing & validation ---
        try:
            destination = request.form.get('destination', '').strip()
            days = max(1, min(int(request.form.get('days', 3)), 30))
            start_date_str = request.form.get('start_date') or datetime.today().strftime('%Y-%m-%d')
            num_travelers = max(1, int(request.form.get('num_travelers', 1)))
            budget_amount = float(request.form.get('budget_amount', 15000))
        except (ValueError, TypeError):
            flash('Please enter valid numbers for days, travelers and budget.', 'danger')
            return redirect(url_for('planner'))

        if not destination:
            flash('Please enter a destination.', 'danger')
            return redirect(url_for('planner'))
        if budget_amount <= 0:
            flash('Budget must be greater than zero.', 'danger')
            return redirect(url_for('planner'))

        currency = request.form.get('currency', 'INR')
        travel_type = request.form.get('travel_type', 'Friends Group')
        interests = ", ".join(request.form.getlist('interests')) or "Sightseeing"
        dietary = ", ".join(request.form.getlist('dietary')) or "Flexible"
        accessibility = ", ".join(request.form.getlist('accessibility')) or "None"
        is_student = 1 if 'is_student' in request.form else 0

        # --- AI-POWERED BUDGET FEASIBILITY CHECK (skipped silently if the AI is unavailable) ---
        try:
            check_prompt = (
                f"Act as a strict financial travel auditor. Can a trip for {num_travelers} traveler(s) "
                f"to {destination} for {days} days be reasonably done on a total budget of "
                f"{currency} {budget_amount}? Answer ONLY 'YES' or 'NO'."
            )
            check_text = call_gemini(check_prompt, json_mode=False, timeout=10, retries=0).strip().upper()
            if check_text.startswith("NO"):
                flash(f"⚠️ AI Budget Check Failed: {currency} {budget_amount} is severely underfunded for a {days}-day trip to {destination} for {num_travelers} traveler(s). Please increase your budget!", "danger")
                return redirect(url_for('planner'))
        except Exception as e:
            print("Budget check skipped:", e)

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

            if parsed_data.get('_fallback'):
                flash('The AI service was unavailable, so a basic template itinerary was created. Use "Optimize" on the trip page to regenerate it.', 'warning')

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
    except (json.JSONDecodeError, TypeError):
        itinerary_data = {}

    # Repair trips saved earlier with empty/partial data so sections never render blank.
    template = build_itinerary_template(
        trip['destination'], trip['days'], trip['start_date'], trip['end_date'],
        trip['num_travelers'], trip['budget_amount'], trip['currency']
    )
    itinerary_data = normalize_itinerary(itinerary_data, template)

    weather_data = get_weather(trip['destination'])
    return render_template('itinerary.html', trip=trip, data=itinerary_data, weather=weather_data)


@app.route('/api/trip/<int:trip_id>/notes', methods=['POST'])
def update_notes(trip_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

    req_data = request.get_json(silent=True) or {}
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

    req_data = request.get_json(silent=True) or {}
    prompt_modifier = req_data.get('prompt', '')

    try:
        # allow_fallback=False: if the AI fails, keep the existing itinerary instead of overwriting it.
        raw_json = generate_ai_itinerary(
            trip['destination'], trip['days'], trip['start_date'], trip['end_date'], trip['num_travelers'],
            trip['budget_amount'], trip['currency'], trip['travel_type'],
            trip['interests'], trip['dietary_prefs'], trip['accessibility_prefs'], bool(trip['is_student']),
            custom_prompt=f"REFINEMENT/MODIFICATION REQUEST: {prompt_modifier}",
            allow_fallback=False
        )
        checklist_json = json.dumps(json.loads(raw_json).get('packing_checklist', []))
        db.cursor().execute(
            'UPDATE trips SET itinerary_json = ?, checklist = ? WHERE id = ? AND user_id = ?',
            (raw_json, checklist_json, trip_id, session['user_id'])
        )
        db.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': f'AI regeneration failed, your existing itinerary was kept. ({e})'}), 502


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
