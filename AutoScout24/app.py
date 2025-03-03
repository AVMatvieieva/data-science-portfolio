import streamlit as st
st.set_page_config(layout="wide")

import joblib
import pandas as pd
from datetime import datetime

# Upload model, scaler andLabelEncoders
model = joblib.load('model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🚗 Auto-Preis Vorhersage")

# Make 2 columns
col1, col2 = st.columns(2)

with col1:
    mileage = st.number_input("Kilometerstand: ", min_value=0, max_value=1000000, step = 100)
    make = st.selectbox("Marke: ",['BMW', 'Volkswagen', 'SEAT', 'Renault', 'Peugeot', 'Toyota', 'Opel' 'Mazda','Ford', 'Mercedes-Benz', 'Chevrolet', 'Audi', 'Fiat', 'Kia', 'Dacia', 'MINI',
    'Hyundai', 'Skoda', 'Citroen', 'Infiniti', 'Suzuki', 'SsangYong', 'smart',
    'Cupra', 'Volvo', 'Jaguar', 'Porsche', 'Nissan', 'Honda', 'Mitsubishi', 'Lexus',
    'Jeep', 'Maserati', 'Bentley', 'Land', 'Alfa', 'Subaru', 'Dodge', 'Microcar',
    'Lamborghini', 'Lada', 'Tesla', 'Chrysler', 'McLaren', 'Aston', 'Rolls-Royce',
    'Lancia', 'Abarth', 'DS', 'Daihatsu', 'Ligier', 'Ferrari', 'Aixam', 'Zhidou',
    'Morgan', 'Maybach', 'RAM', 'Alpina', 'Polestar', 'Brilliance', 'Piaggio',
    'FISKER', 'Others', 'Cadillac', 'Iveco', 'Isuzu', 'Corvette', 'Baic', 'DFSK',
    'Estrima', 'Alpine'])
    model_name = st.selectbox("Model: ", ['Golf', 'Exeo', 'Megane', '308', 'Auris', 'Scenic', 'Zafira', '3', 'Transit', 'Meriva', 'Orlando', 'A4', 'Polo', 'Espace', 'Grand Espace', 'Sedici', 'Corsa', 'Picanto', 'Duster', 'Cross Touran', 'Beetle', 'Golf Cabriolet', 'Astra', '118', 'Focus', 'One D Countryman', 'Cooper Clubman', 'B 180', 'Insignia', 'One', '320', 'iX35', 'Adam', 'B-Max', 'Golf Variant', 'Touran', '114', 'Galaxy', 'Leon', 'A1', 'Trax', 'Verso', 'Golf Plus', 'Fiesta', 'Vivaro', 'Yaris', 'iX20', 'up!', 'Rapid/Spaceback', '116', 'Passat Variant', "Ceed / cee'd", 'Fabia', 'Ibiza', 'C-Max', 'Superb', 'C4 Cactus', '208', '316', 'Swift', 'Korando', 'Sandero', '2', 'Sorento', 'GLC 220', 'A6', 'E 400', 'forTwo', 'Mii', 'Citigo', 'Aygo', 'Punto', 'i10', 'forFour', 'A6 allroad', 'A4 allroad', 'GLC 250', 'Q2', 'Tiguan', 'Ateca', 'Amarok', 'Q5', 'SQ5', 'Q7', 'M2', 'XC90', 'X3', 'X2', 'T-Roc', 'Vito', 'Kuga', 'Karoq', 'GLA 250', 'Alhambra', 'RS Q3', 'TT RS', 'F-Type', 'Macan', 'Touareg', 'A7', 'S5', 'T6 California', 'Q8', 'Panda', '508', 'Qashqai', 'Civic', 'Passat', 'Avensis', 'Octavia', 'Altea', 'Mondeo', 'Jetta', 'i40', 'Grand C4 Picasso', '2008', 'SX4', 'Jazz', 'i20', 'Yeti', 'Roomster', 'Venga', 'DS3', 'Captur', 'Clio', 'C4', 'Rio', 'Scudo', 'Pulsar', 'Caddy', 'ZOE', 'C3', 'Citan', 'Micra', 'Twingo', 'Ka/Ka+', 'Space Star', '500', 'Eclipse Cross', 'Grandland X', 'XF', 'Passat Alltrack', 'Combo', 'E 300', 'E 450', 'RX 450h', 'XC60', 'RS3', 'CLS 350', 'F-Pace', 'Arona', 'EcoSport', 'Puma', 'i30', 'Note', '206', '207', 'Doblo', "Ceed SW / cee'd SW", 'Grande Punto', 'Logan', 'Punto Evo', 'Mokka', '330', 'S-Max', 'Transit Custom', '530', 'GLA 180', 'A 160', '430', '520', '535', 'T6 Multivan', 'E 220', 'C 250', 'C 300', 'A 45 AMG', 'S3', 'GLC 350', 'V90 Cross Country', 'V 220', 'A5', 'Edge', 'V60', 'E 200', 'Golf GTI', 'Ioniq', 'A3', 'C 220', 'Camry', 'Tucson', 'Talisman', 'Optima', 'T6.1 Transporter', 'CX-30', 'Crossland X', 'Tourneo Custom', 'Compass', 'Formentor', '5008', 'Ranger', 'Spacetourer', 'A 180', 'Cruze', '108', 'Phaeton', 'GLE 350', 'S4', 'GranTurismo', 'Continental', 'Zafira Tourer', 'Kona', 'Cooper D', '218', 'C-HR', 'Cooper S', 'A8', 'SQ7', 'RS', 'E 53 AMG', 'Cayenne', 'GLS 500', 'XC40', 'Kodiaq', 'CX-5', 'Arteon', 'Sharan', '3008', 'ID.3', 'V 300', 'V 250', 'T6.1 Multivan', 'S60', 'T6.1 California', 'GLC 43 AMG', 'NV200', 'Grand C-Max', 'Veloster', 'S40', 'Antara', 'Crafter', 'Scirocco', 'V40', '220', '500C', 'Partner', 'Carens', 'iOn', 'C1', 'Express', 'Stonic', 'Colt', 'Spark', 'Transit Connect', 'Berlingo', 'New Panda', 'i3', 'Rover Range Rover Evoque', 'Rover Discovery Sport', '335', 'V60 Cross Country', 'CLA 200', 'Cooper S Countryman', 'Cascada', 'SX4 S-Cross', 'Cooper', '318', 'One Countryman', 'One Cabrio', '6', 'Prius', 'Scala', 'GT86', 'RAV 4', 'Corolla', 'Q3', 'X1', 'Navara', 'Tarraco', 'Grand Scenic', '5', 'Verso-S', 'V50', 'Altea XL', 'C3 Picasso', 'One D', 'Tivoli', 'Tipo', 'Transit Courier', 'Karl', '500L', 'MX-30', 'Juke', 'Captiva', 'B 160', 'Romeo Giulietta', 'Cooper D Clubman', 'Kangoo', 'C5', 'Mokka X', 'CX-3', 'E 250', 'V90', 'X6', '730', '550', 'Rover Discovery', '650', 'Koleos', 'A 250', '525', 'C 200', 'Kadjar', 'DS5', '4008', 'Golf Sportsvan', 'Q70', 'Sprinter', 'Toledo', 'E 350', 'CLA 250', 'RS5', 'e-tron', 'RS4', 'Challenger', 'Rover Range Rover', 'RS6', 'RS7', 'Polo GTI', 'Freemont', 'Qashqai+2', '325', 'C4 Aircross', 'C-Elysée', 'Due', 'T-Cross', 'EQC 400', 'GLA 45 AMG', 'Rover Range Rover Velar', 'GLE 400', 'S 350', 'Aveo', 'Splash', 'Aventador', 'Rover Range Rover Sport', 'C 43 AMG', 'C 63 AMG', 'E 43 AMG', 'X6 M', 'Wrangler', 'B 200', 'Soul', 'S90', 'Santa Fe', 'Zafira Life', 'Alto', 'Stinger', '540', 'GLC 300', 'GLC 63 AMG', 'Sportage', 'IS 300', 'AMG GT', 'X7 M', 'S 63 AMG', 'Dokker', 'GLB 200', 'Cayman', 'Panamera', 'R8', 'Vitara', 'Eos', 'Jumper', '120', 'Tourneo Courier', 'C4 Picasso', '420', '140', 'John Cooper Works Countryman', 'S6', 'V40 Cross Country', 'Grand Cherokee', 'Supra', 'Mustang', '340', 'Rover Defender', 'NX 300', 'Enyaq', 'M8', '911', 'Tourneo Connect', 'XV', 'C 180', 'Accord', 'DS4', 'ASX', 'Kalina', 'C 400', 'Model 3', 'GLC 200', 'Outlander', 'CR-V', 'Grand Voyager', '216', 'Celerio', 'X2 M', 'RS Q8', 'Passat CC', 'CC', 'Laguna', 'Cross Golf', 'Renegade', 'Jumpy', 'E 63 AMG', 'Model X', '991', '570S', 'Martin Vantage', 'Ghost', 'Levante', 'M5', 'E-Pace', 'T6 Caravelle', 'GLB 250', 'Bipper', 'John Cooper Works', 'MX-5', 'A 200', '418', 'TT', '135', "ProCeed / pro_cee'd", 'XCeed', 'Cooper SE', 'Cooper Countryman', 'Cooper D Cabrio', 'Modus', 'Fusion', 'Romeo MiTo', 'C30', 'Master'])
    fuel = st.selectbox("Treibstoff: ", ['Gasoline', 'Electric/Gasoline', 'Diesel', '-/- (Fuel)', 'Electric', 'Electric/Diesel', 'LPG', 'CNG', 'Others', 'Hydrogen', 'Ethanol'])

with col2:
    gear = st.selectbox("Getriebe: ",['Manual', 'Automatic', 'Semi-automatic'])
    offertype = st.selectbox("Angebotsart:", ['Used', 'Demonstration', "Employee's car", 'Pre-registered', 'New'])
    hp = st.number_input("Leistung (PS):", min_value=50, max_value=1000, step=10)
    year = st.slider("Baujahr:", min_value=2000, max_value=2024, step=1)

# Calculate the year
age = datetime.now().year - year

if st.sidebar.button("Preis vorhersagen"):
    # Data transform
    input_data = pd.DataFrame([[mileage, make, model_name, fuel, gear, offertype, hp, age]],
    columns = ["mileage", "make", "model", "fuel", "gear", "offerType", "hp", "age"])
    for col in ["make", "model", "fuel", "gear", "offerType"]:
        
        value = input_data[col][0]
        if value in label_encoders[col].classes_:
            input_data[col] = label_encoders[col].transform([value])[0]
            
        else:
            st.error(f"❌ Fehler: Unbekannter Wert in {col}")
            st.stop()
            
    # Transform num value
    input_data[["mileage", "hp", "age"]] = scaler.transform(input_data[["mileage", "hp", "age"]]) 
    
    # Predict
    predicted_price = model.predict(input_data)[0]
    
    st.sidebar.success(f"📌 Geschätzter Preis: **{predicted_price:,.2f} €**")     