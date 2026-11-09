import os
import re
import json
import sqlite3
import requests
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, g
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "dev-only-change-me")
DATABASE = os.getenv("DATABASE_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), 'travelgenie.db'))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# Optional: set GEMINI_USE_SEARCH=1 to let Gemini ground hotels/restaurants/prices in Google Search.
# (Grounding cannot be combined with JSON mime-type mode, so JSON is extracted from the text instead.)
USE_SEARCH_GROUNDING = os.getenv("GEMINI_USE_SEARCH", "0") == "1"

MAX_DAYS = 14
MAX_TRAVELERS = 20


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


# --- Gemini helpers ---
def call_gemini(prompt, json_mode=True, grounded=False, timeout=90, temperature=0.8, max_tokens=24000):
    """Call Gemini and return the response text. Raises RuntimeError on any failure."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set on the server.")

    gen_cfg = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
        # No hidden "thinking" tokens eating into the output budget
        "thinkingConfig": {"thinkingBudget": 0},
    }
    if json_mode and not grounded:
        gen_cfg["responseMimeType"] = "application/json"

    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen_cfg}
    if grounded:
        payload["tools"] = [{"google_search": {}}]

    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    resp = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=timeout)
    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError(f"Gemini returned a non-JSON response (HTTP {resp.status_code}).")

    if resp.status_code != 200:
        msg = data.get("error", {}).get("message", f"HTTP {resp.status_code}")
        raise RuntimeError(f"Gemini API error: {msg}")

    candidates = data.get("candidates") or []
    if not candidates:
        reason = data.get("promptFeedback", {}).get("blockReason", "unknown")
        raise RuntimeError(f"Gemini returned no candidates (reason: {reason}).")

    cand = candidates[0]
    if cand.get("finishReason") == "MAX_TOKENS":
        raise RuntimeError("Gemini response was cut off (MAX_TOKENS).")

    parts = cand.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text


def extract_json(text):
    """Parse JSON from model output, tolerating markdown fences and stray prose."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def build_day_dates(start_date, days):
    """Return a list of 'Mon DD, YYYY' strings, one per day (falls back to 'Day N')."""
    try:
        start = datetime.strptime(start_date, '%b %d, %Y')
        return [(start + timedelta(days=i)).strftime('%b %d, %Y') for i in range(days)]
    except Exception:
        return [f"Day {i + 1}" for i in range(days)]


BUDGET_KEYS = ["hotel", "food", "transport", "tickets", "shopping", "emergency"]
DEFAULT_RATIOS = {"hotel": 0.4, "food": 0.25, "transport": 0.15, "tickets": 0.1, "shopping": 0.05, "emergency": 0.05}


def _normalize_budget(raw, total):
    """Make sure the breakdown has all categories and adds up exactly to the total budget."""
    raw = raw if isinstance(raw, dict) else {}
    vals = {}
    for k in BUDGET_KEYS:
        try:
            vals[k] = max(float(raw.get(k, 0)), 0.0)
        except (TypeError, ValueError):
            vals[k] = 0.0

    if sum(vals.values()) <= 0:
        vals = {k: total * r for k, r in DEFAULT_RATIOS.items()}

    scale = total / sum(vals.values())
    out = {k: int(round(v * scale)) for k, v in vals.items()}
    # fix rounding drift on the biggest category
    drift = int(round(total)) - sum(out.values())
    biggest = max(out, key=out.get)
    out[biggest] += drift
    out["total"] = int(round(total))
    return out


def normalize_itinerary(data, destination, days, start_date, end_date, num_travelers,
                        budget_amount, currency, per_person_budget):
    """Validate Gemini output and enforce authoritative values / safe structure for the templates."""
    if not isinstance(data, dict):
        raise ValueError("Gemini output was not a JSON object.")

    day_list = data.get("day_by_day")
    if not isinstance(day_list, list) or len(day_list) != days:
        got = len(day_list) if isinstance(day_list, list) else 0
        raise ValueError(f"Expected {days} day-by-day entries, got {got}.")

    dates = build_day_dates(start_date, days)
    for i, day in enumerate(day_list):
        if not isinstance(day, dict):
            raise ValueError("Malformed day entry.")
        day["day"] = i + 1
        day["date"] = dates[i]
        for key in ("theme", "morning", "lunch", "afternoon", "evening", "estimated_cost"):
            day.setdefault(key, "")

    # Authoritative values from the user's form (never trust the model for these)
    data.update({
        "destination": destination,
        "days": days,
        "start_date": start_date,
        "end_date": end_date,
        "num_travelers": num_travelers,
        "currency": currency,
        "total_budget": budget_amount,
        "per_person_budget": per_person_budget,
    })
    data["budget_breakdown"] = _normalize_budget(data.get("budget_breakdown"), budget_amount)
    data.setdefault("estimated_total_cost", f"{currency} {budget_amount:,.0f}")

    for key in ("latitude", "longitude"):
        try:
            data[key] = float(data.get(key))
        except (TypeError, ValueError):
            data[key] = 0.0

    # Structural defaults so templates never hit a missing key
    data.setdefault("currency_info", {})
    data["currency_info"].setdefault("currency_name", currency)
    data["currency_info"].setdefault("exchange_rate", "")
    data.setdefault("detailed_weather", {})
    for key in ("current_temp", "forecast", "humidity", "wind_speed", "rain_chance", "best_time_to_travel"):
        data["detailed_weather"].setdefault(key, "")
    data.setdefault("safety_and_scam_warnings", [])
    data.setdefault("transit_guide", {})
    for key in ("flight_cost", "train_cost", "bus_cost", "nearest_airport", "nearest_railway", "local_transport"):
        data["transit_guide"].setdefault(key, "")
    data.setdefault("ai_suggestions", {})
    for key in ("best_month", "local_dish_must_try", "pro_tip"):
        data["ai_suggestions"].setdefault(key, "")
    data.setdefault("hotel_recommendations", [])
    data.setdefault("restaurant_suggestions", {})
    for key in ("breakfast", "lunch", "dinner", "famous_local_dishes", "budget_restaurants", "premium_restaurants"):
        data["restaurant_suggestions"].setdefault(key, [])
    data.setdefault("map_pins", [])

    checklist = data.get("packing_checklist")
    if not isinstance(checklist, list):
        checklist = []
    for item in checklist:
        if isinstance(item, dict):
            item["checked"] = False
    data["packing_checklist"] = checklist
    return data


def get_weather(destination, fallback=None):
    """Live weather if WEATHER_API_KEY works; otherwise use the Gemini-generated seasonal weather."""
    fb = fallback or {}
    fallback_weather = {
        "current_temp": fb.get("current_temp", "N/A"),
        "forecast": fb.get("forecast", "N/A"),
        "humidity": fb.get("humidity", "N/A"),
        "wind_speed": fb.get("wind_speed", "N/A"),
        "rain_chance": fb.get("rain_chance", "N/A"),
        "best_time": fb.get("best_time_to_travel", "N/A"),
    }
    if not WEATHER_API_KEY:
        return fallback_weather
    try:
        res = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": destination, "appid": WEATHER_API_KEY, "units": "metric"},
            timeout=5,
        ).json()
        if str(res.get("cod")) == "200":
            return {
                "current_temp": f"{round(res['main']['temp'])} °C",
                "forecast": res['weather'][0]['description'].capitalize(),
                "humidity": f"{res['main']['humidity']}%",
                "wind_speed": f"{round(res['wind']['speed'] * 3.6)} km/h",
                "rain_chance": f"{res.get('clouds', {}).get('all', 0)}% cloud cover",
                "best_time": fallback_weather["best_time"],
            }
    except Exception:
        pass
    return fallback_weather


def generate_ai_itinerary(destination, days, start_date, end_date, num_travelers, budget_amount,
                          currency, travel_type, interests, dietary, accessibility,
                          is_student=False, custom_prompt=""):
    """Ask Gemini for a fully destination-specific itinerary and return it as a JSON string."""
    per_person_budget = round(budget_amount / max(num_travelers, 1), 2)
    per_day_budget = round(budget_amount / max(days, 1))
    day_dates = build_day_dates(start_date, days)

    student_instruction = ""
    if is_student:
        student_instruction = """
[STUDENT MODE]
The travelers are students carrying valid College IDs. Explicitly mention student-ID discounts wherever they
genuinely exist (museums, monuments, amusement parks, metro/rail concessions, youth passes). Prefer hostels, dorms,
budget guesthouses, street food and public transport, and factor discounted student ticket prices into the costs.
Do NOT invent a discount if you are not reasonably sure one exists - say "check for student concession" instead.
"""

    refinement = f"\n[USER REFINEMENT REQUEST - apply this while keeping everything else consistent]\n{custom_prompt}\n" if custom_prompt else ""

    # The schema uses <angle-bracket> instructions (not sample values) so the model can't just copy placeholders.
    schema = {
        "destination": destination,
        "days": days,
        "start_date": start_date,
        "end_date": end_date,
        "num_travelers": num_travelers,
        "currency": currency,
        "total_budget": budget_amount,
        "per_person_budget": per_person_budget,
        "estimated_total_cost": f"<{currency} amount you realistically expect this trip to cost, not above the budget>",
        "currency_info": {
            "currency_name": f"<full name of {currency}>",
            "exchange_rate": f"<approximate 1 USD = X {currency}, marked as approximate>",
        },
        "detailed_weather": {
            "current_temp": "<typical temperature at the destination during the travel dates, e.g. '24 °C'>",
            "forecast": "<typical conditions for those dates>",
            "humidity": "<typical % for those dates>",
            "wind_speed": "<typical km/h>",
            "rain_chance": "<typical chance of rain for those dates>",
            "best_time_to_travel": "<best months/season for this specific destination>",
        },
        "budget_breakdown": {
            "hotel": 0, "food": 0, "transport": 0, "tickets": 0, "shopping": 0, "emergency": 0,
            "total": budget_amount,
        },
        "latitude": 0.0,
        "longitude": 0.0,
        "safety_and_scam_warnings": ["<4-6 destination-specific safety tips and real, known local scams>"],
        "transit_guide": {
            "flight_cost": f"<typical one-way per-person flight cost to the destination in {currency}, or 'N/A'>",
            "train_cost": f"<typical per-person train cost in {currency}, or 'N/A'>",
            "bus_cost": f"<typical per-person bus cost in {currency}, or 'N/A'>",
            "nearest_airport": "<real airport name and IATA code>",
            "nearest_railway": "<real railway station name>",
            "local_transport": "<the real local transport options and typical fares in this destination>",
        },
        "ai_suggestions": {
            "best_month": "<best months to visit and why>",
            "local_dish_must_try": "<2-3 real signature dishes of this destination>",
            "pro_tip": "<one specific, genuinely useful insider tip for this destination>",
        },
        "hotel_recommendations": [
            {
                "name": "<real hotel/hostel/stay name in the destination>",
                "rating": "<e.g. '4.3 ⭐'>",
                "price": f"<realistic price per night in {currency}>/night",
                "distance": "<distance from the main area/landmark>",
                "amenities": "<comma-separated real amenities>",
            }
        ],
        "restaurant_suggestions": {
            "breakfast": [{"name": "<real place>", "dish": "<real dish>", "cost": f"<{currency} per person>", "rating": "<x.x ⭐>"}],
            "lunch": [{"name": "<real place>", "dish": "<real dish>", "cost": f"<{currency} per person>", "rating": "<x.x ⭐>"}],
            "dinner": [{"name": "<real place>", "dish": "<real dish>", "cost": f"<{currency} per person>", "rating": "<x.x ⭐>"}],
            "famous_local_dishes": ["<real local dish or street food>"],
            "budget_restaurants": [{"name": "<real place>", "cost": f"<{currency} per person>", "rating": "<x.x ⭐>"}],
            "premium_restaurants": [{"name": "<real place>", "cost": f"<{currency} per person>", "rating": "<x.x ⭐>"}],
        },
        "map_pins": [{"name": "<real attraction/place from the itinerary>", "lat": 0.0, "lng": 0.0, "category": "<Attraction | Food | Hotel | Viewpoint | Market | Activity>"}],
        "day_by_day": [
            {
                "day": 1,
                "date": day_dates[0],
                "theme": "<short unique theme for this day>",
                "morning": "<named real place + what to do + rough timing>",
                "lunch": "<named real restaurant/stall + what to eat>",
                "afternoon": "<named real place/activity>",
                "evening": "<named real place/activity, incl. dinner suggestion>",
                "estimated_cost": f"<{currency} estimate for this day, for the whole group>",
            }
        ],
        "packing_checklist": [{"item": "<item>", "category": "<Essential | Documents | Electronics | Health | Clothing | Gear | Weather | Personal Care>", "checked": False}],
    }

    prompt = f"""
You are a meticulous local travel expert for TravelGenie AI. Plan a trip to {destination}.

TRIP DETAILS
- Duration: {days} day(s), {start_date} to {end_date}
- Day dates (in order): {", ".join(day_dates)}
- Travelers: {num_travelers} person(s) | Travel style: {travel_type}
- Total budget: {currency} {budget_amount:,.0f} (about {currency} {per_person_budget:,.0f} per person, about {currency} {per_day_budget:,} per day for the group)
- Interests: {interests}
- Dietary preferences: {dietary}
- Accessibility / group needs: {accessibility}
{student_instruction}{refinement}

CONTENT RULES (very important)
1. day_by_day: output EXACTLY {days} entries, one per date above. Every day must have a distinct theme and DIFFERENT named
   places - never reuse an attraction, restaurant, or sentence across days. Name specific real attractions, neighbourhoods,
   markets, viewpoints and activities that exist in {destination} and match the interests. Group nearby places on the same
   day to minimise travel. Day 1 should allow for arrival and the last day for departure. Vary the pace (e.g. one lighter day).
   Adjust activities to the season/weather of the travel dates and to the dietary and accessibility needs above.
2. hotel_recommendations: 3 real, currently-operating stays in {destination} spanning budget / mid-range / comfort, each with
   a realistic nightly price in {currency} that fits the budget for {num_travelers} traveler(s). Use real local price levels.
3. restaurant_suggestions: real, named restaurants/cafes/street-food spots in {destination}, 2 each for breakfast, lunch and
   dinner, 4-6 famous_local_dishes, 2-3 budget_restaurants and 2 premium_restaurants. Respect the dietary preferences.
   Costs are per person in {currency}. Only state ratings you are reasonably confident about; otherwise use a modest estimate.
4. ai_suggestions: destination-accurate best months, genuine regional delicacies, and a specific insider pro_tip (not generic).
5. budget_breakdown: allocate the full budget across hotel, food, transport, tickets, shopping, emergency in a way that
   reflects THIS destination, trip length, travel style and group size (do not use a fixed split). The six categories must add
   up to total_budget. Make the day-wise estimated_cost values consistent with this breakdown.
6. safety_and_scam_warnings: 4-6 specific warnings for {destination} (real local scams, areas, transport, weather or
   health risks, women/solo/group advice where relevant). No generic filler.
7. transit_guide: real nearest airport (with IATA code) and railway station, with realistic fares in {currency}.
8. latitude/longitude: the real coordinates of {destination}. map_pins: 6-10 real places from the itinerary with accurate
   coordinates (decimal degrees) covering attractions, food spots and the recommended hotel area.
9. packing_checklist: 10-14 items tailored to {destination}, its weather in the travel dates, the planned activities, and
   the traveler type. "checked" must be false.
10. If you are unsure whether a specific place still exists or what its exact price is, prefer a well-known real place and
    give a rounded realistic price rather than inventing a name.

OUTPUT FORMAT
Return ONLY one valid JSON object (no markdown fences, no commentary) with exactly the structure below. Replace every
<instruction> with real content and replace every 0 / 0.0 with real numbers. Arrays may contain more items than shown
(day_by_day must have exactly {days}). Keep all string values on a single line and escape any double quotes.

{json.dumps(schema, ensure_ascii=False, indent=2)}
"""

    last_error = None
    for attempt in range(2):  # one retry on malformed / incomplete output
        try:
            text = call_gemini(prompt, json_mode=True, grounded=USE_SEARCH_GROUNDING,
                               temperature=0.8 if attempt == 0 else 0.5)
            data = extract_json(text)
            data = normalize_itinerary(
                data, destination, days, start_date, end_date, num_travelers,
                budget_amount, currency, per_person_budget
            )
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            last_error = e
            print(f"Itinerary generation attempt {attempt + 1} failed: {e}")

    raise RuntimeError(f"Could not generate a valid itinerary: {last_error}")


def check_budget_feasible(destination, days, num_travelers, currency, budget_amount):
    """Returns False only if Gemini clearly says the budget is unrealistic. Fails open on errors."""
    prompt = (
        f"Act as a realistic travel budget auditor. Can a trip for {num_travelers} traveler(s) to {destination} "
        f"for {days} day(s) reasonably be done on a total budget of {currency} {budget_amount:,.0f} "
        f"(basic but safe stays, local food, local transport, main sights)? Answer ONLY 'YES' or 'NO'."
    )
    try:
        answer = call_gemini(prompt, json_mode=False, timeout=15, temperature=0, max_tokens=16).strip().upper()
        return not answer.startswith("NO")
    except Exception as e:
        print("Budget check error:", e)
        return True


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
        destination = request.form.get('destination', '').strip()
        if not destination:
            flash('Please enter a destination.', 'warning')
            return redirect(url_for('planner'))

        try:
            days = max(1, min(int(request.form.get('days', 3)), MAX_DAYS))
            num_travelers = max(1, min(int(request.form.get('num_travelers', 1)), MAX_TRAVELERS))
            budget_amount = float(request.form.get('budget_amount', 15000))
            if budget_amount <= 0:
                raise ValueError
        except ValueError:
            flash('Please enter valid numbers for days, travelers and budget.', 'warning')
            return redirect(url_for('planner'))

        start_date_str = request.form.get('start_date', datetime.today().strftime('%Y-%m-%d'))
        currency = request.form.get('currency', 'INR')
        travel_type = request.form.get('travel_type', 'Friends Group')
        interests = ", ".join(request.form.getlist('interests')) or "Sightseeing"
        dietary = ", ".join(request.form.getlist('dietary')) or "Flexible"
        accessibility = ", ".join(request.form.getlist('accessibility')) or "None"
        is_student = 1 if 'is_student' in request.form else 0

        # --- AI-POWERED BUDGET FEASIBILITY CHECK ---
        if not check_budget_feasible(destination, days, num_travelers, currency, budget_amount):
            flash(f"⚠️ AI Budget Check Failed: {currency} {budget_amount:,.0f} looks severely underfunded for a "
                  f"{days}-day trip to {destination} for {num_travelers} traveler(s). Please increase your budget!", "danger")
            return redirect(url_for('planner'))

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

    itinerary_data = json.loads(trip['itinerary_json'])
    weather_data = get_weather(trip['destination'], fallback=itinerary_data.get('detailed_weather'))
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
    prompt_modifier = req_data.get('prompt', '').strip()

    try:
        raw_json = generate_ai_itinerary(
            trip['destination'], trip['days'], trip['start_date'], trip['end_date'], trip['num_travelers'],
            trip['budget_amount'], trip['currency'], trip['travel_type'],
            trip['interests'], trip['dietary_prefs'], trip['accessibility_prefs'], bool(trip['is_student']),
            custom_prompt=prompt_modifier
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
