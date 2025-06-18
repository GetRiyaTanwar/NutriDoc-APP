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

# Set page configuration
st.set_page_config(page_title="Welcome to NutriDoc APP", layout="wide")

# --- API Key Input ---
api_key = st.sidebar.text_input("Enter your Google API Key", type="password")

if api_key:
    genai.configure(api_key=api_key)
    st.session_state.api_key_configured = True
else:
    st.session_state.api_key_configured = False
    st.sidebar.warning("Please enter your Google API Key to use the app.")

# Cache data to optimize resource usage
@st.cache_data
def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content)).convert("RGBA")
    else:
        raise Exception(f"Failed to download image from URL. Status code: {response.status_code}")

# Make an image circular
def make_circular(image):
    np_image = np.array(image)
    h, w = np_image.shape[:2]

    # Create an alpha mask
    alpha = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(alpha)
    draw.pieslice([0, 0, w, h], 0, 360, fill=255)
    np_alpha = np.array(alpha)

    # Add alpha layer to the image
    np_image = np.dstack((np_image[:, :, :3], np_alpha))
    return Image.fromarray(np_image)

# Get response from Gemini 1.5 Pro
def get_gemini_response(input_prompt, image, retries=3, delay=5):
    if not st.session_state.api_key_configured:
        st.error("API Key not configured. Please enter your Google API Key.")
        return None

    # Initialize Gemini 1.5 Pro
    model = genai.GenerativeModel("gemini-1.5-pro")
    attempt = 0
    while attempt < retries:
        try:
            response = model.generate_content([input_prompt, image[0]])
            if not response.text:
                if response.safety_ratings:
                    st.warning(f"Response blocked due to safety ratings: {response.safety_ratings}")
                else:
                    st.warning("No valid response text received.")
                raise Exception("No valid response text received.")
            return response.text
        except Exception as e:
            st.error(f"Attempt {attempt + 1} failed: {e}")
            attempt += 1
            time.sleep(delay * attempt)
    raise Exception("All retry attempts failed. Please check the service status or contact support.")

# Process uploaded image
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Parse nutrition response
def parse_nutrition_response(response_text):
    patterns = [
        r'(\w+)\s*-\s*(\d+)\s*calories',
        r'(\w+)\s*:\s*(\d+)\s*calories',
    ]

    nutrition_data = {}
    for pattern in patterns:
        matches = re.findall(pattern, response_text)
        for match in matches:
            nutrition_data[match[0]] = int(match[1])
    
    return nutrition_data

# Plot a bar chart of nutritional data
@st.cache_data
def plot_bar_chart(nutrition_data):
    if not nutrition_data:
        st.warning("No nutritional data found to plot the bar chart.")
        return None

    labels = list(nutrition_data.keys())
    sizes = list(nutrition_data.values())
    colors = ['green' if calories < 100 else 'red' for calories in sizes]

    fig, ax = plt.subplots()
    ax.bar(labels, sizes, color=colors)

    ax.set_ylabel('Calories')
    ax.set_title('Nutritional Content by Food Item')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    return fig

# Calculate BMI
def calculate_bmi(weight, height_cm):
    if height_cm == 0:
        return 0  # Avoid division by zero
    return weight / (height_cm / 100) ** 2

# Convert height from inches to cm
def height_in_inches_to_cm(inches):
    return inches * 2.54

# Generate diet advice
def get_diet_advice(bmi, health_issues, nutrition_data):
    advice = []
    if bmi < 18.5 and bmi != 0: # Check for non-zero BMI before giving advice
        advice.append("You are underweight. Focus on calorie-dense foods and healthy fats.")
    elif bmi >= 25:
        advice.append("You are overweight. Focus on a balanced diet with portion control.")
    elif 18.5 <= bmi < 25:
        advice.append("Your BMI is healthy. Maintain a balanced diet.")
    
    if "Heart Issue" in health_issues:
        advice.append("Limit saturated fats and cholesterol. Focus on lean proteins and fiber-rich foods.")
    if "Diabetes" in health_issues:
        advice.append("Monitor carbohydrate intake. Choose whole grains and avoid added sugars.")
    if "Hypertension" in health_issues:
        advice.append("Limit sodium intake. Focus on fruits, vegetables, and low-fat dairy products.")
    if "PCOD" in health_issues:
        advice.append("Focus on whole foods and avoid processed foods. Balance carbohydrates with proteins.")
    
    if nutrition_data:
        for item, calories in nutrition_data.items():
            if calories > 200:
                advice.append(f"Consider reducing calorie intake from {item} as it's high in calories.")
            elif calories > 100:
                pass # Moderate calories, no specific warning
            else:
                advice.append(f"{item} is a good low-calorie option.")
    
    if not advice:
        advice.append("No specific diet advice generated based on the provided information.")

    return advice

# Display the app
logo_url = "https://drive.google.com/uc?export=download&id=16716w0N7Yuh5X4l-uMncEkScmoEcMNK1"
@st.cache_data
def get_logo_image(url):
    try:
        logo_image = download_image(url)
        logo_image = make_circular(logo_image)
        # Save to a temporary file path that Streamlit can access
        temp_path = os.path.join("/tmp", "logo_circle.png")
        if not os.path.exists("/tmp"):
            os.makedirs("/tmp")
        logo_image.save(temp_path)
        return temp_path
    except Exception as e:
        st.warning(f"Could not load logo image: {e}. Please ensure the URL is correct and accessible.")
        return None

logo_image_path = get_logo_image(logo_url)
if logo_image_path:
    st.image(logo_image_path, width=200)

st.markdown("# Welcome to NutriDoc APP", unsafe_allow_html=True)
st.markdown("### Enter your profile information")
sex = st.radio("Sex", ("Male", "Female"))
health_issues = st.multiselect("Health Issues", ["Heart Issue", "Diabetes", "Hypertension", "PCOD", "None"])
dietary_preference = st.radio("Dietary Preference", ("Vegetarian", "Non-Vegetarian", "Vegan"))

st.markdown("### Enter your details to calculate BMI")
col1, col2 = st.columns(2)
with col1:
    inches = st.number_input("Height (in inches)", min_value=0.0, format="%.2f")
with col2:
    weight = st.number_input("Weight (in kg)", min_value=0.0, format="%.2f")

if inches > 0 and weight > 0:
    height_cm = height_in_inches_to_cm(inches)
    bmi = calculate_bmi(weight, height_cm)
    st.markdown(f"### Your BMI: {bmi:.2f}")

    input_prompt = f"""
    You are a nutrition expert. Identify food items from the image and calculate total calories for each identified food item.
    Also, provide a summary of the nutritional content and any potential dietary recommendations based on the identified foods and the user's profile.
    
    User Profile:
    - Sex: {sex}
    - Health Issues: {", ".join(health_issues) if health_issues else "None"}
    - Dietary Preference: {dietary_preference}
    - BMI: {bmi:.2f}

    Format the calorie information clearly, for example: "FoodItem - XXX calories".
    """

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Image"):
            if st.session_state.api_key_configured:
                with st.spinner("Analyzing image..."):
                    image_data = input_image_setup(uploaded_file)
                    try:
                        response_text = get_gemini_response(input_prompt, image_data)
                        if response_text:
                            st.markdown("### Analysis Response:")
                            st.write(response_text)

                            nutrition_data = parse_nutrition_response(response_text)
                            if nutrition_data:
                                fig = plot_bar_chart(nutrition_data)
                                if fig:
                                    st.pyplot(fig)

                                advice = get_diet_advice(bmi, health_issues, nutrition_data)
                                st.markdown("### Diet Advice:")
                                for adv in advice:
                                    st.info(adv) # Using st.info for better readability of advice points
                            else:
                                st.warning("No specific nutritional data found in the response to plot or provide detailed advice.")
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
            else:
                st.error("Please enter your Google API Key to analyze the image.")
else:
    st.info("Please enter valid height and weight to calculate BMI and proceed with analysis.")
