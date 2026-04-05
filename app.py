import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# Page Config
st.set_page_config(page_title="Kernel SVM Interactive Dashboard", layout="wide")

st.title("🛡️ Kernel SVM: Social Network Ads")
st.markdown("Analyze how Age and Salary influence purchasing decisions using a Radial Basis Function (RBF) Kernel.")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    # Ensure the CSV is in the same directory
    df = pd.read_csv('Social_Network_Ads.csv')
    return df

try:
    dataset = load_data()
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Sidebar Controls
    st.sidebar.header("Model Hyperparameters")
    test_size = st.sidebar.slider("Test Set Size", 0.1, 0.5, 0.25)
    c_param = st.sidebar.number_input("C (Regularization)", value=1.0, min_value=0.01)
    
    # --- PROCESSING ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    classifier = SVC(kernel='rbf', C=c_param, random_state=0)
    classifier.fit(X_train_scaled, y_train)

    y_pred = classifier.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    # --- METRICS ---
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Accuracy", f"{acc*100:.2f}%")
    with col2:
        st.write("**Confusion Matrix:**")
        st.write(confusion_matrix(y_test, y_pred))

    # --- PLOTLY INTERACTIVE BOUNDARY ---
    st.subheader("Decision Boundary Visualization")
    
    def plot_decision_boundary(X_data, y_data, title):
        # Create a mesh grid
        h = 0.5  # step size in mesh
        x_min, x_max = X_data[:, 0].min() - 5, X_data[:, 0].max() + 5
        y_min, y_max = X_data[:, 1].min() - 5000, X_data[:, 1].max() + 5000
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h*200))
        
        # Predict over mesh
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = classifier.predict(sc.transform(grid_points))
        Z = Z.reshape(xx.shape)

        fig = go.Figure()

        # Add Contour for decision boundary
        fig.add_trace(go.Contour(
            x=np.arange(x_min, x_max, h),
            y=np.arange(y_min, y_max, h*200),
            z=Z,
            showscale=False,
            colorscale=[[0, '#FA8072'], [1, '#1E90FF']],
            opacity=0.4,
            hoverinfo='skip'
        ))

        # Add Scatter points
        for i, label in enumerate(np.unique(y_data)):
            mask = (y_data == label)
            fig.add_trace(go.Scatter(
                x=X_data[mask, 0],
                y=X_data[mask, 1],
                mode='markers',
                name=f'Class {label}',
                marker=dict(
                    color='#FA8072' if label == 0 else '#1E90FF',
                    size=10,
                    line=dict(width=1, color='Black')
                )
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Age",
            yaxis_title="Estimated Salary",
            template="plotly_white",
            height=600
        )
        return fig

    tab1, tab2 = st.tabs(["Training Set", "Test Set"])
    
    with tab1:
        st.plotly_chart(plot_decision_boundary(X_train, y_train, "Kernel SVM (Training set)"), use_container_width=True)
    
    with tab2:
        st.plotly_chart(plot_decision_boundary(X_test, y_test, "Kernel SVM (Test set)"), use_container_width=True)

    # --- PREDICTION TOOL ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Live Prediction")
    input_age = st.sidebar.slider("Age", 18, 60, 30)
    input_salary = st.sidebar.number_input("Salary", value=87000, step=1000)
    
    prediction = classifier.predict(sc.transform([[input_age, input_salary]]))
    result = "Purchased" if prediction[0] == 1 else "Not Purchased"
    color = "green" if prediction[0] == 1 else "red"
    st.sidebar.markdown(f"**Prediction:** :{color}[{result}]")

except FileNotFoundError:
    st.error("Error: 'Social_Network_Ads.csv' not found. Please ensure the file is in the root directory.")
