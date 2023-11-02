from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
module = pickle.load(open('mobile.pkl', 'rb'))

# Brand mapping
brand_mapping = {
    "OnePlus": 1,
    "Realme": 2,
    "Apple": 3,
    "LG": 4,
    "Samsung": 5,
    "Asus": 6,
    "Xiaomi": 7,
    "Oppo": 8,
    "Huawei": 9,
    "Google": 10,
    "Nokia": 11,
    "HTC": 12,
    "Motorola": 13,
    "Honor": 14,
    "Yu": 15,
    "Poco": 16,
    "Vivo": 17,
    "Nubia": 18,
    "Black Shark": 19,
    "Infinix": 20,
    "Lenovo": 21,
    "Sony": 22,
    "Jio": 23,
    "Coolpad": 24,
    "Micromax": 25,
    "Smartron": 26,
    "LeEco": 27,
    "BlackBerry": 28,
    "Gionee": 29,
    "Meizu": 30,
    "Panasonic": 31,
    "Tecno": 32,
    "InFocus": 33,
    "Itel": 34,
    "10.or": 35,
    "Lava": 36,
    "Cat": 37,
    "Lyf": 38,
    "Intex": 39,
    "Xolo": 40,
    "Acer": 41,
    "Phicomm": 42,
    "Karbonn": 43,
    "Spice": 44,
    "iVoomi": 45,
    "Kult": 46,
    "Nuu Mobile": 47,
    "Ziox": 48,
    "Zopo": 49,
}

app = Flask(__name__, template_folder="templates")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Map brand to its numeric value
            brand = request.form['a']
            brand_numeric = brand_mapping.get(brand, 0)  # 0 for unknown brand

            # Extract other input data
            data2 = int(request.form['b'])
            data3 = int(request.form['c'])
            data4 = int(request.form['d'])
            data5 = int(request.form['e'])

            arr = np.array([[brand_numeric, data2, data3, data4, data5]])
            pred = module.predict(arr)

            return render_template('home.html', prediction=pred, highlight=True)
        except ValueError:
            return render_template('home.html', error="Invalid input. Please enter valid integers.")

if __name__ == "__main__":
    app.run(debug=True)
