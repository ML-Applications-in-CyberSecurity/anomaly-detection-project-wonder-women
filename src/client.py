import csv
import socket
import json
import pandas as pd
import joblib
import os
from together import Together
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore", message="Field \"model_id\" has conflict with protected namespace")



HOST = 'localhost'
PORT = 9999

model = joblib.load("../anomaly_model.joblib")

def pre_process_data(data):
    df = pd.DataFrame([data])
    df_processed = pd.get_dummies(df, columns=['protocol'], drop_first=True)
    expected_columns = ['src_port', 'dst_port', 'packet_size', 'duration_ms', 'protocol_UDP']
    for col in expected_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0
    df_processed = df_processed[expected_columns]
    return np.array(df_processed)

# CSV file initialization
csv_file = 'anomalies.csv'
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['src_port', 'dst_port', 'packet_size', 'duration_ms', 'protocol', 'description'])

normal_samples = []
anomaly_samples = []


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

                if prediction[0] == -1:
                    anomaly_samples.append(processed_data[0])  # add as 1D array
                else:
                    normal_samples.append(processed_data[0])  # add as 1D array


                if prediction[0] == -1:
                    print(f"Anomaly detected in data: {data}")
                    #api_key = os.getenv('TOGETHER_API_KEY')
                    api_key= "9730b2822edd7fb9bf3b453a4650e76f1b7225f025dc2b5e21d078c82b29722f"
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

                        # Record anomaly to CSV
                        with open(csv_file, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([
                                data.get('src_port'),
                                data.get('dst_port'),
                                data.get('packet_size'),
                                data.get('duration_ms'),
                                data.get('protocol'),
                                description
                            ])

                    except Exception as e:
                        print(f"Error connecting to Together AI API: {e}")
                else:
                    print(f"Normal data: {data}")

            except json.JSONDecodeError:
                print("Error decoding JSON.")

    import pandas as pd

    # Combine samples
    X = normal_samples + anomaly_samples
    y = [0] * len(normal_samples) + [1] * len(anomaly_samples)  # 0=normal, 1=anomaly

    if len(X) > 0:  # avoid errors if no data was collected
        # Reduce to 2D with PCA
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)

        # Create DataFrame for seaborn
        df_vis = pd.DataFrame({
            'PC1': X_reduced[:, 0],
            'PC2': X_reduced[:, 1],
            'Label': ['Normal' if label == 0 else 'Anomaly' for label in y]
        })

        # Plot
        plt.figure(figsize=(10, 7))
        sns.scatterplot(
            data=df_vis,
            x='PC1', y='PC2',
            hue='Label',
            palette={'Normal': 'blue', 'Anomaly': 'red'},
            alpha=0.7
        )
        plt.title('PCA Visualization of Normal vs. Anomalous Data')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.show()
    else:
        print("No data collected for visualization.")

