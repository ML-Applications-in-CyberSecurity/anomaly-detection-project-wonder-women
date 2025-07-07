import socket
import json
import pandas as pd
import joblib
import os
from together import Together
import numpy as np

HOST = 'localhost'
PORT = 9999

model = joblib.load("../anomaly_model.joblib")

def pre_process_data(data):
    #DataFrame
    df = pd.DataFrame([data])
    #  One-Hot Encoding 
    df_processed = pd.get_dummies(df, columns=['protocol'], drop_first=True)
   
    expected_columns = ['src_port', 'dst_port', 'packet_size', 'duration_ms', 'protocol_UDP']
    for col in expected_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0 #empty colums = 0
    # sort
    df_processed = df_processed[expected_columns]
    # NumPy
    return np.array(df_processed)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    buffer = ""
    print("Client connected to server.\n")

    while True:
        chunk = s.recv(1024).decode()
        if not chunk:
            break
        buffer += chunk

        while '\n' in buffer:
            line, buffer = buffer.split('\n', 1)
            try:
                data = json.loads(line)
                print(f'Data Received:\n{data}\n')

                processed_data = pre_process_data(data)

                prediction = model.predict(processed_data)

                # analysis
                if prediction[0] == -1:
                    print(f"Anomaly detected in data: {data}")
                    #API Together AI
                    api_key = os.getenv('TOGETHER_API_KEY')#متغیر محلی
                    client = Together(api_key=api_key)
                    messages = [
                        {"role": "system", "content": "You are an expert in cybersecurity anomaly detection."},
                        {"role": "user", "content": f"Analyze the network traffic data: {data}. Identify the type of anomaly and suggest a possible cause in a concise manner."}
                    ]
                    try:
                        response = client.chat.completions.create(
                            model="meta-llama/Llama-3-70b-chat-hf",
                            messages=messages,
                            stream=False,
                        )
                        description = response.choices[0].message.content
                        print(f"\n🚨 Anomaly Detected!\nData: {data}\nDescription: {description}\n")
                    except Exception as e:
                        print(f"Error connecting to Together AI API: {e}")
                else:
                    print(f"Normal data: {data}")

            except json.JSONDecodeError:
                print("Error decoding JSON.")
