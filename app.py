import streamlit as st
import google.generativeai as genai
import os
import matplotlib.pyplot as plt
import re
from PIL import Image, ImageDraw
import time
import requests
import numpy as np
from io import BytesIO

# Set page configuration first
st.set_page_config(page_title="Welcome to NutriDoc APP", layout="wide")

# Constants and Session State Initialization (same as before)
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW = 60
MIN_REQUEST_INTERVAL = 1
CACHE_TTL = 300
MAX_CACHE_SIZE = 100

if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = False
if 'last_request_time' not in st.session_state:
    st.session_state.last_request_time = 0
if 'request_count' not in st.session_state:
    st.session_state.request_count = 0
if 'cached_results' not in st.session_state:
    st.session_state.cached_results = {}

# --- Helper Functions (Caching, Rate Limiting, Image Processing) ---
# These functions remain the same as in your original code.
def cache_response(prompt, response_text):
    if len(st.session_state.cached_results) >= MAX_CACHE_SIZE:
        oldest_key = min(st.session_state.cached_results, key=lambda k: st.session_state.cached_results[k]['timestamp'])
        del st.session_state.cached_results[oldest_key]
    st.session_state.cached_results[prompt] = {'response': response_text, 'timestamp': time.time()}

def get_cached_response(prompt):
    if prompt not in st.session_state.cached_results: return None
    cached = st.session_state.cached_results[prompt]
    if time.time() - cached['timestamp'] > CACHE_TTL:
        del st.session_state.cached_results[prompt]
        return None
    return cached['response']

def check_rate_limit():
    current_time = time.time()
    if st.session_state.last_request_time and current_time - st.session_state.last_request_time < MIN_REQUEST_INTERVAL:
        st.warning(f"Please wait {MIN_REQUEST_INTERVAL - (current_time - st.session_state.last_request_time):.1f}s.")
        return False
    if st.session_state.request_count >= RATE_LIMIT_REQUESTS and current_time - st.session_state.last_request_time < RATE_LIMIT_WINDOW:
        remaining = RATE_LIMIT_WINDOW - (current_time - st.session_state.last_request_time)
        st.warning(f"Rate limit reached. Wait {remaining/60:.1f} minutes.")
        return False
    return True

def convert_google_drive_url(url):
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if match: return f"https://drive.google.com/uc?export=download&id={match.group(1)}"
    return url

@st.cache_data(ttl=CACHE_TTL)
def download_image(url):
    response = requests.get(url)
    if response.status_code == 200: return Image.open(BytesIO(response.content)).convert("RGBA")
    else: raise Exception(f"Failed to download image. Status: {response.status_code}")

def make_circular(image):
    np_image = np.array(image)
    h, w = np_image.shape[:2]
    alpha = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(alpha)
    draw.pieslice([0, 0, w, h], 0, 360, fill=255)
    np_alpha = np.array(alpha)
    np_image = np.dstack((np_image[:, :, :3], np_alpha))
    return Image.fromarray(np_image)

@st.cache_data
def get_logo_image(url):
    try:
        url = convert_google_drive_url(url)
        logo_image = download_image(url)
        logo_image = make_circular(logo_image)
        temp_path = "/tmp/logo_circle.png"
        if not os.path.exists("/tmp"): os.makedirs("/tmp")
        logo_image.save(temp_path)
        return temp_path
    except Exception as e:
        st.warning(f"Could not load logo: {e}")
        return None

def input_image_setup(uploaded_file):
    if uploaded_file: return [{"mime_type": uploaded_file.type, "data": uploaded_file.getvalue()}]
    else: raise FileNotFoundError("No file uploaded")
# --- END Helper Functions ---


# --- NEW AND IMPROVED PARSING AND PLOTTING ---
def parse_nutrition_response(response_text):
    """
    Parses a markdown table from the AI response to extract detailed nutritional data.
    """
    nutrition_data = {}
    lines = response_text.split('\n')
    table_started = False
    for line in lines:
        if '|' in line and '---' not in line and 'Item' not in line:
            table_started = True
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) == 6:
                try:
                    item_name = parts[0]
                    nutrition_data[item_name] = {
                        "Protein (g)": float(parts[1]),
                        "Carbohydrates (g)": float(parts[2]),
                        "Fats (g)": float(parts[3]),
                        "Sugar (g)": float(parts[4]),
                        "Total Calories": float(parts[5]),
                    }
                except (ValueError, IndexError):
                    continue
        elif table_started and not line.strip():
            break
    return nutrition_data

@st.cache_data
def plot_nutrition_chart(nutrition_data):
    """
    Creates a stacked bar chart to show the macronutrient breakdown.
    """
    if not nutrition_data:
        st.warning("No nutritional data found to plot.")
        return None

    labels = list(nutrition_data.keys())
    proteins = [data['Protein (g)'] for data in nutrition_data.values()]
    carbs = [data['Carbohydrates (g)'] for data in nutrition_data.values()]
    fats = [data['Fats (g)'] for data in nutrition_data.values()]
    
    fig, ax = plt.subplots()
    width = 0.6

    ax.bar(labels, carbs, width, label='Carbohydrates (g)', color='#1f77b4')
    ax.bar(labels, proteins, width, bottom=carbs, label='Protein (g)', color='#2ca02c')
    carbs_plus_proteins = [c + p for c, p in zip(carbs, proteins)]
    ax.bar(labels, fats, width, bottom=carbs_plus_proteins, label='Fats (g)', color='#ff7f0e')

    ax.set_ylabel('Grams (g)')
    ax.set_title('Macronutrient Breakdown by Food Item')
    ax.legend()
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    plt.tight_layout() # Adjust layout to make room for labels
    
    return fig
# --- END NEW FUNCTIONS ---

# --- Gemini API and BMI/Advice Functions (BMI/Advice functions are unchanged) ---
def get_gemini_response(input_prompt, image, retries=3, delay=5):
    if not st.session_state.api_key_configured:
        st.error("API Key not configured.")
        return None
    cached_response = get_cached_response(input_prompt)
    if cached_response:
        return cached_response
    if not check_rate_limit():
        return None
    model = genai.GenerativeModel("gemini-1.5-flash")
    attempt = 0
    while attempt < retries:
        try:
            response = model.generate_content([input_prompt, image[0]])
            if not response.text:
                raise Exception(f"No valid response text. Safety ratings: {response.safety_ratings}")
            cache_response(input_prompt, response.text)
            st.session_state.last_request_time = time.time()
            st.session_state.request_count += 1
            return response.text
        except Exception as e:
            if "quota" in str(e).lower() or "limit" in str(e).lower():
                st.error("API quota limit reached.")
                return None
            st.error(f"Attempt {attempt + 1} failed: {e}")
            attempt += 1
            time.sleep(delay * attempt)
    st.error("All retry attempts failed.")
    return None

def calculate_bmi(weight, height_cm):
    if height_cm == 0: return 0
    return weight / (height_cm / 100) ** 2

def height_in_inches_to_cm(inches):
    return inches * 2.54

def get_diet_advice(bmi, health_issues):
    advice = []
    if bmi < 18.5 and bmi != 0: advice.append("You are underweight. Focus on calorie-dense foods.")
    elif bmi >= 25: advice.append("You are overweight. Focus on portion control and high-fiber foods.")
    elif 18.5 <= bmi < 25: advice.append("Your BMI is healthy. Maintain a balanced diet.")
    if "Heart Issue" in health_issues: advice.append("Limit saturated fats and sodium.")
    if "Diabetes" in health_issues: advice.append("Monitor carbohydrate intake and focus on low-glycemic foods.")
    if "Hypertension" in health_issues: advice.append("Limit sodium intake.")
    if "PCOD" in health_issues: advice.append("Avoid processed foods and refined sugars; balance carbs with proteins.")
    if not advice: advice.append("Maintain a balanced diet and regular exercise.")
    return advice
# --- END ---


# --- UI AND MAIN APP LOGIC ---
st.sidebar.title("Configuration")
st.sidebar.markdown("""
**How to Get Your Google API Key:**
1.  Go to [Google AI Studio](https://aistudio.google.com/).
2.  Sign in and click **"Get API key"**.
3.  Create and copy your new API key.
""")
api_key = st.sidebar.text_input("Enter your Google API Key", type="password", key="api_key_input")

if api_key:
    try:
        genai.configure(api_key=api_key)
        st.session_state.api_key_configured = True
    except Exception as e:
        st.session_state.api_key_configured = False
        st.sidebar.error(f"Invalid API Key: {e}")
else:
    st.sidebar.warning("Please enter your Google API Key.")

logo_url = "https://drive.google.com/file/d/1KY6LI2vyHx3zUTSF8-VP4_zJssk2ck5Z/view?usp=sharing"
logo_image_path = get_logo_image(logo_url)
if logo_image_path:
    st.image(logo_image_path, width=150)

st.markdown("# Welcome to the NutriDoc APP")
st.markdown("### Step 1: Enter your profile information")
sex = st.radio("Sex", ("Male", "Female", "Prefer not to say"))
health_issues = st.multiselect("Any existing health issues?", ["Heart Issue", "Diabetes", "Hypertension", "PCOD", "None"])
dietary_preference = st.radio("Dietary Preference", ("Vegetarian", "Non-Vegetarian", "Vegan"))

st.markdown("### Step 2: Enter your details to calculate BMI")
col1, col2 = st.columns(2)
with col1:
    inches = st.number_input("Height (in inches)", min_value=0.0, format="%.2f")
with col2:
    weight = st.number_input("Weight (in kg)", min_value=0.0, format="%.2f")

if inches > 0 and weight > 0:
    height_cm = height_in_inches_to_cm(inches)
    bmi = calculate_bmi(weight, height_cm)
    st.metric(label="Your BMI is", value=f"{bmi:.2f}")

    # THIS IS THE NEW, MORE DETAILED PROMPT
    input_prompt = f"""
    You are an expert nutritionist. Your task is to analyze the food items in the provided image and provide a detailed nutritional breakdown.

    **Instructions:**
    1.  Identify each distinct food item in the meal.
    2.  For each item, provide your best estimate for Protein (g), Carbohydrates (g), Fats (g), Sugar (g), and Total Calories. **Do not use ranges; provide a single estimated number for each value.**
    3.  Present this information ONLY in a markdown table with the following exact headers: `Item | Protein (g) | Carbohydrates (g) | Fats (g) | Sugar (g) | Total Calories`
    4.  After the table, provide a concise summary and dietary recommendations based on this nutritional data and the user's profile below.

    **User Profile:**
    - Sex: {sex}
    - Health Issues: {", ".join(health_issues) if health_issues else "None"}
    - Dietary Preference: {dietary_preference}
    - BMI: {bmi:.2f}

    **Example of the required markdown table format:**
    | Item | Protein (g) | Carbohydrates (g) | Fats (g) | Sugar (g) | Total Calories |
    |---|---|---|---|---|---|
    | Rice | 5 | 45 | 1 | 0 | 200 |
    | Sambar | 8 | 15 | 5 | 3 | 150 |

    Begin your response with the markdown table.
    """

    st.markdown("### Step 3: Upload an image of your meal")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Meal", use_column_width=True)

        if st.button("Analyze My Meal"):
            if st.session_state.api_key_configured:
                with st.spinner("Our AI nutritionist is analyzing your meal..."):
                    image_data = input_image_setup(uploaded_file)
                    try:
                        response_text = get_gemini_response(input_prompt, image_data)
                        if response_text:
                            st.markdown("---")
                            st.markdown("### Nutritional Analysis")
                            
                            # The full text response from the AI is still valuable
                            st.write(response_text) 
                            
                            # Use the new parser
                            nutrition_data = parse_nutrition_response(response_text)

                            if nutrition_data:
                                # Use the new plotter
                                fig = plot_nutrition_chart(nutrition_data)
                                if fig:
                                    st.pyplot(fig)

                                st.markdown("### Personalized Diet Advice")
                                advice = get_diet_advice(bmi, health_issues)
                                for adv in advice:
                                    st.info(f"💡 {adv}")
                            else:
                                st.warning("Could not extract structured nutritional data from the analysis. The AI's response might not have followed the required format.")
                    except Exception as e:
                        st.error(f"An error occurred during analysis: {e}")
            else:
                st.error("Please enter a valid Google API Key in the sidebar to analyze the image.")
else:
    st.info("Please enter your height and weight to calculate your BMI and get started.")