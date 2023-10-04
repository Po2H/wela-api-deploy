from flask import Flask, jsonify, request, render_template
import pymongo
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    options = [
        'โรงแรม รีสอร์ทและห้องชุด', 'เกสต์เฮ้าส์', 'ที่พักสัมผัสวัฒนธรรมชนบท', 
        'กิจกรรมที่พักแรมระยะสั้นอื่น ๆ ซึ่งมิได้จัดประเภทไว้ในที่อื่น', 
        'ลานตั้งค่ายพักแรม ที่จอดรถพ่วงและที่ตั้งที่พักแบบเคลื่อนที่', 
        'การบริการห้องพักหรือที่พักอาศัยสำหรับนักเรียน/นักศึกษา', 
        'การบริการที่พักแรมประเภทอื่นๆซึ่งมิได้จัดประเภทไว้ในที่อื่น', 
        'การบริการด้านอาหารในภัตตาคาร/ร้านอาหาร', 'การบริการด้านอาหารบนแผงลอยและตลาด', 
        'การบริการด้านอาหารโดยร้านอาหารแบบเคลื่อนที่', 'การบริการด้านการจัดเลี้ยง', 
        'การบริการอาหารสำหรับกิจการขนส่ง', 'การดำเนินงานของโรงอาหาร', 
        'การบริการด้านอาหารอื่นๆซึ่งมิได้จัดประเภทไว้ในที่อื่น', 
        'การบริการด้านเครื่องดื่มที่มีแอลกอฮอล์เป็นหลักในร้าน', 
        'การบริการด้านเครื่องดื่มที่ไม่มีแอลกอฮอล์เป็นหลักในร้าน', 
        'การบริการด้านเครื่องดื่มบนแผงลอยและตลาด', 'การบริการด้านเครื่องดื่มโดยร้านเคลื่อนที่'
    ]
    return render_template('index.html', options=options)

# Load the model and the encoder
model = joblib.load('ml.pkl')
encoder = joblib.load('encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(data)  # print received data

    # Extracting values from received data
    cat = data['categorical_data']
    profit = data['numerical_data']

    # Custom input data
    custom_value = pd.DataFrame({'business_type': [cat], 'profit_margin': [profit]})

    # Encode the custom input data
    custom_input_encoded = encoder.transform(custom_value)
    
    # Make predictions
    custom_output = model.predict(custom_input_encoded)

    print(custom_output)
    
    # Returning the response as JSON with the actual prediction
    return jsonify({
        'prediction': custom_output[0]
    })

if __name__ == "__main__":
    app.run(debug=True)
