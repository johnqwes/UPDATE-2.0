import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import pickle
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin  # Import BaseEstimator and RegressorMixin
from sklearn.linear_model import LinearRegression
import plotly.express as px
import os
import io
import base64
import pyrebase
import time
import pickle
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

stacked_model_filename = 'stacked_model.pkl'
dropout_model_filename = 'stacked_model_drop.pkl'
grad_model_filename = 'stacked_model_grad.pkl'

class forecasting_regressor(BaseEstimator, RegressorMixin):
    def __init__(self, change_probability=0.5, max_change=5):
        self.change_probability = change_probability
        self.max_change = max_change
        self.last_value = None

    def fit(self, X, y):
        if len(y) < 1:
            raise ValueError("Empty target variable y provided.")
        # Store all the historical data
        self.historical_data = list(y)
        # Use the last value in y to predict the next year's value
        self.last_value = y[-1]
        return self

    def predict(self, X):
        if self.last_value is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        # Determine how many years of historical data to consider
        if self.last_value <= 3:  # If the last value is within 3 years, consider all historical data
            combined_data = self.historical_data
        else:  # If the last value is more than 3 years, consider only the last 5 years
            combined_data = self.historical_data[-5:]
        
        # Calculate the predicted value for the next year based on historical data
        predicted_value = np.mean(combined_data) + np.random.randint(-self.max_change, self.max_change + 1)
        return np.full(len(X), predicted_value)  # Return an array with shape (n_samples,)

    def score(self, X, y):
        # For simplicity, you can return a constant score
        return 0  # Modify as needed


st.set_page_config(page_title="LU Student Trend", page_icon=":bar_chart:", layout="wide")

DATA_FILE = 'data.csv'
MODEL_FILE = 'linear_regression_model.pkl'

# Function to load the model from a pickle file
def load_model(stacked_model_filename):
    with open(stacked_model_filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Function to save the model to a pickle file
def save_model(model, stacked_model_filename):
    with open(stacked_model_filename, 'wb') as file:
        pickle.dump(model, file)

def retrain_model(data, stacked_model_filename):
    # Load the existing model
    stacked_model = load_model(stacked_model_filename)

    # Split the data into features and target variable
    X = data.drop(columns=['Number of Enrollees', 'Program Name', 'Number of Dropout', 'Number of Graduates'])  # Features
    y = data['Number of Enrollees']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define base models
    base_models = [
        ('linear', LinearRegression()),
        ('rf', RandomForestRegressor()),
        ('dt', DecisionTreeRegressor()),
        ('svm', SVR(kernel='rbf')),
        ('xgb', XGBRegressor()),
        ('knn', KNeighborsRegressor(2)),
        # ('gb', GradientBoostingRegressor()),  # Gradient Boosting Regressor
        ('mlp', MLPRegressor()),  # Multi-layer Perceptron Regressor
    ]

    # Initialize stacking regressor using the base models and a meta-regressor
    stacked_model = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())

    # Train the stacking model
    stacked_model.fit(X_train, y_train)

    # Save the updated model
    save_model(stacked_model, stacked_model_filename)

    # Return the updated model
    return stacked_model

# Define the file path for the data
DATA_FILE = 'data.csv'

# Define the file path for the stacked model
stacked_model_filename = 'stacked_model.pkl'

# Load data
data = pd.read_csv(DATA_FILE)


def retrain_dropout_model(data, dropout_model_filename):
    # Load the existing dropout model
    dropout_model = load_model(dropout_model_filename)

    # Split the data into features and target variable
    X = data.drop(columns=['Number of Enrollees', 'Program Name', 'Number of Dropout', 'Number of Graduates'])  # Features
    y = data['Number of Dropout']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define base models for dropout prediction
    dropout_base_models = [
        ('linear', LinearRegression()),
        ('rf', RandomForestRegressor()),
        ('dt', DecisionTreeRegressor()),
        ('svm', SVR(kernel='rbf')),
        ('xgb', XGBRegressor()),
        ('knn', KNeighborsRegressor(2)),
        # ('gb', GradientBoostingRegressor()),  # Gradient Boosting Regressor
        ('mlp', MLPRegressor()),  # Multi-layer Perceptron Regressor
    ]

    # Initialize stacking regressor using the base models and a meta-regressor
    dropout_stacked_model = StackingRegressor(estimators=dropout_base_models, final_estimator=LinearRegression())

    # Train the dropout stacking model
    dropout_stacked_model.fit(X_train, y_train)

    # Save the updated dropout model
    save_model(dropout_stacked_model, dropout_model_filename)

    # Return the updated dropout model
    return dropout_stacked_model

# Define the file path for the data
DATA_FILE = 'data.csv'

# Define the file path for the dropout model
dropout_model_filename = 'stacked_model_drop.pkl'

# Load data
data = pd.read_csv(DATA_FILE)


def retrain_grad_model(data, grad_model_filename):
    # Load the existing dropout model
    grad_model = load_model(grad_model_filename)

    # Split the data into features and target variable
    X = data.drop(columns=['Program Name', 'Number of Dropout', 'Number of Graduates'])  # Features
    y = data['Number of Graduates']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define base models for dropout prediction
    grad_base_models = [
        ('linear', LinearRegression()),
        ('rf', RandomForestRegressor()),
        ('dt', DecisionTreeRegressor()),
        ('svm', SVR(kernel='rbf')),
        ('xgb', XGBRegressor()),
        ('knn', KNeighborsRegressor(2)),
        # ('gb', GradientBoostingRegressor()),  # Gradient Boosting Regressor
        ('mlp', MLPRegressor()),  # Multi-layer Perceptron Regressor
    ]

    # Initialize stacking regressor using the base models and a meta-regressor
    grad_stacked_model = StackingRegressor(estimators=grad_base_models, final_estimator=LinearRegression())

    # Train the dropout stacking model
    grad_stacked_model.fit(X_train, y_train)

    # Save the updated dropout model
    save_model(grad_stacked_model, grad_model_filename)

    # Return the updated dropout model
    return grad_stacked_model

# Define the file path for the data
DATA_FILE = 'data.csv'

# Define the file path for the dropout model
grad_model_filename = 'stacked_model_grad.pkl'

# Load data
data = pd.read_csv(DATA_FILE)



def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def save_data(data):
    data.to_csv(DATA_FILE, index=False)

def set_background(background_image_path):
    background_image_ext = 'png'  # Modify the extension if needed
    encoded_image = base64.b64encode(open(background_image_path, "rb").read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('data:image/{background_image_ext};base64,{encoded_image}');
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Function to load and save data
def load_data():
    return pd.read_csv(DATA_FILE)



img = get_img_as_base64("green.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://scontent.fmnl30-1.fna.fbcdn.net/v/t1.15752-9/409412298_1453175752285633_1675291559160145762_n.png?_nc_cat=103&ccb=1-7&_nc_sid=8cd0a2&_nc_eui2=AeFh2vXyR7tyUo6LV54YZG4RJce81NeXzEQlx7zU15fMRHcvim4_4fhhLlNzMLGydc7kHaGuEsqrdK-PMbuGwor6&_nc_ohc=Osj1_e0B9NEAX-vcouK&_nc_ht=scontent.fmnl30-1.fna&oh=03_AdSdriG4aAEmp2p7CJ6Ch8dOs3lzAqmqPDG-jBJzCnWFbQ&oe=65F02D6F");
background-size: cover;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


# Firebase initialization
firebaseConfig = {
    "apiKey": "AIzaSyCM-2QTRJPaDaPZnXRR5aSqkZRAIPFLaQg",
    "authDomain": "streamlit-cde99.firebaseapp.com",
    "databaseURL": "https://streamlit-cde99-default-rtdb.firebaseio.com/",
    "projectId": "streamlit-cde99",
    "storageBucket": "streamlit-cde99.appspot.com",
    "messagingSenderId": "651067602910",
    "appId": "1:651067602910:web:b98e16ab9aa07fd4000f4f",
    "measurementId": "G-XEQM0PYDYZ",
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database()

def authenticate(email, password):
    try:
        # Check if the email is in declined accounts
        declined_accounts = db.child("declined_accounts").get().val()
        if declined_accounts and email in declined_accounts:
            st.error("Your account request has been declined.")
            return None

        # Check if the email/password match any user in approved_users
        approved_users = db.child("approved_users").get().val()
        if approved_users:
            for user_email, user_data in approved_users.items():
                if email == user_data.get("email") and password == user_data.get("password"):
                    # Return the authenticated user data
                    return user_data

        # If not found in approved_users or declined_accounts, attempt authentication using Firebase Authentication
        login = auth.sign_in_with_email_and_password(email, password)
        return login
    except Exception as e:
        error_message = str(e)
        if "INVALID_PASSWORD" in error_message or "INVALID_EMAIL" in error_message:
            st.error("Invalid email or password.")
        return None
    
    
def sidebar_bg(side_bg):
    try:
        # Get the file extension
        _, side_bg_ext = os.path.splitext(side_bg)
        # Read the image file and encode it as base64
        with open(side_bg, "rb") as img_file:
            img_data = img_file.read()
            encoded_img = base64.b64encode(img_data).decode()
        # Set the background style for the sidebar
        st.markdown(
            f"""
            <style>
            [data-testid="stSidebar"] > div:first-child {{
                background: url(data:image/{side_bg_ext};base64,{encoded_img}) !important;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        st.error(f"Image file '{side_bg}' not found.")
    except Exception as e:
        st.error(f"Error: {e}")
    
side_bg = "static/image1.jpg"
sidebar_bg(side_bg)
    
# Function to send a password reset email
def send_password_reset_email(email):
    if not email:
        st.toast("Please, input email.")
        time.sleep(2)
        return

    try:
        auth.send_password_reset_email(email)
        st.toast("Password reset email sent.")
        time.sleep(2)
    except Exception as e:
        error_message = str(e)
        if "INVALID_EMAIL" in error_message:
            st.toast("Invalid email address. Please check your email.")
            time.sleep(2)
        elif "MISSING_EMAIL" in error_message:
            st.toast("Missing email address. Please enter your email.")
            time.sleep(2)
        else:
            st.toast(f"Error sending password reset email: {e}")
            time.sleep(2)


def signup(email, password):
    try:
        # Store user details in pending sign-up requests
        db.child("pending_signups").child(email.replace(".", ",")).set({"email": email, "password": password})
        st.success("Sign up successful. Please wait for admin approval.")
    except Exception as e:
        st.error(f"Error signing up: {e}")

def validate_email(email):
    if "@" not in email or "." not in email:
        return False
    return True

def validate_password(password):
    if len(password) < 6:
        return False
    return True

def approve_signup(email):
    try:
        # Get the sign-up request data
        signup_data = db.child("pending_signups").child(email.replace(".", ",")).get().val()
        if signup_data:
            # Move user to approved users list
            db.child("approved_users").child(email.replace(".", ",")).set(signup_data)
            # Delete user from pending sign-ups
            db.child("pending_signups").child(email.replace(".", ",")).remove()
            st.success(f"Account for {email} approved.")
        else:
            st.error("No sign-up request found for the given email.")
    except Exception as e:
        st.error(f"Error approving sign-up request: {e}")

# Function to get pending sign-up requests
def get_pending_signups():
    try:
        pending_signups = db.child("pending_signups").get().val()
        return pending_signups
    except Exception as e:
        st.error(f"Error retrieving pending sign-up requests: {e}")
        return None
    
def get_firebase_db_users():
    try:
        # Retrieve list of users from Firebase Realtime Database
        user_data = db.child("approved_users").get().val()
        if user_data:
            user_emails = [data.get("email") for data in user_data.values()]  # Accessing email using get method
            return user_emails
        else:
            return []
    except Exception as e:
        st.error(f"Error retrieving Firebase Realtime Database users: {e}")
        return []
    
def change_password(email, old_password, new_password):
    try:
        # Authenticate the user using the email and old password from the Realtime Database
        user_data = db.child("approved_users").order_by_child("email").equal_to(email).get()
        if user_data.each():
            for user in user_data.each():
                user_email = user.val().get("email")
                user_password = user.val().get("password")
                if user_email == email and user_password == old_password:
                    # Change password
                    db.child("approved_users").child(user.key()).update({"password": new_password})
                    st.success("Password changed successfully.")
                    return
            # If the email or old password doesn't match
            st.error("Invalid email or old password.")
        else:
            # If the email is not found
            st.error("User not found.")
    except Exception as e:
        st.error(f"Error changing password: {e}")
    
def manage_accounts():
    try:
        # Dropdown menu for account actions
        action = st.selectbox("Select Action:", ["Approval", "Account Deletion", "Change Password"])

        if action == "Approval":
            st.subheader("Approval of Account")
            pending_signups = get_pending_signups()

            if pending_signups:
                st.write("Pending Sign-up Requests:")
                for email, signup_data in pending_signups.items():
                    st.write(f"Email: {email}")
                    approve_button = st.button(f"Approve {email}")
                    decline_button = st.button(f"Decline {email}")
                    if approve_button:
                        approve_signup(email)
                    elif decline_button:
                        decline_signup(email)
            else:
                st.write("No pending sign-up requests.")

        elif action == "Account Deletion":
            st.subheader("Delete Account")
            firebase_db_users = get_firebase_db_users()
            
            email_to_delete = st.selectbox("Select Email Address to Delete:", [""] + firebase_db_users)
            
            if st.button("Delete Account") and email_to_delete:
                delete_account(email_to_delete)

        elif action == "Change Password":
            st.subheader("Change Password")
            email = st.text_input("Email Address")
            old_password = st.text_input("Old Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            if st.button("Change Password"):
                if email and old_password and new_password and confirm_password:
                    if new_password == confirm_password:
                        change_password(email, old_password, new_password)
                    else:
                        st.error("New passwords do not match.")
                else:
                    st.error("Please fill in all fields.")

    except Exception as e:
        st.error(f"Error managing accounts: {e}")

def decline_signup(email):
    try:
        # Remove user details from pending sign-up requests
        db.child("pending_signups").child(email.replace(".", ",")).remove()
        st.success(f"Request from {email} has been declined.")
    except Exception as e:
        st.error(f"Error declining sign-up request: {e}")

def delete_account(email):
    try:
        # Check if the account exists in approved_users
        user_ref = db.child("approved_users").order_by_child("email").equal_to(email).get()
        if user_ref.each():
            # Remove the user from approved_users
            for user in user_ref.each():
                db.child("approved_users").child(user.key()).remove()
            st.success(f"Account for {email} deleted successfully.")
        else:
            st.error("No account found for the given email.")
    except Exception as e:
        st.error(f"Error deleting account: {e}")

def get_current_logged_in_user():
    # This function retrieves the currently logged-in user from Streamlit's session state
    return st.session_state.get('user')


# Function to logout user
def logout():
    # Clear user info from session state
    st.session_state.user = None

# Streamlit app content
def main():
     
     background_image_path = "static/image.jpg"  # Adjust the path accordingly
     set_background(background_image_path)

    # Create a session state object
     if 'user' not in st.session_state:
        st.session_state.user = None

     if st.session_state.user is None:
        st.markdown("<h1 style=' color: #545454;'>LOGIN</h1>", unsafe_allow_html=True)

        st.markdown(
        """
        <style>
        .st-emotion-cache-q8sbsg p {
            color: black;
        }
        .st-emotion-cache-16idsys p{
            color: #545454;
        }
        button.st-emotion-cache-hc3laj.ef3psqc12 {
            background-color: #2ECC71;
            position: relative;
            border: 1px solid black;
            margin: 0;
            color: #fff;
            display: inline-block;
            text-decoration: none;
            text-align: center;
        }
        .st-b2 {
        background-color: white;
        }
            button.st-emotion-cache-13ejsyy.ef3psqc12{
            background-color: #2f9e36;
            color: #fff;
            transition: 0.2s;
            height: 2.5rem;
        }
        div.st-emotion-cache-1wmy9hl.e1f1d6gn0{
            width: 325px;
            height: 470px;
            background-color: #f0f0f0;
            border: 1px solid #ccc;
            margin: 20px;
            padding: 10px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            position: relative;
            left: 30px;
            border: 3px solid #73AD21;
            border-radius: 2rem;
            margin-top: 90px;
        }
        .st-bo{
            width: 300px;
        }
        .st-emotion-cache-10trblm{
            font-size: 25px;
            text-align: center;
            margin-right: 20px;
        }
        .st-emotion-cache-1vbkxwb p{
            font-size: 12px;
            text-align: center;
        }
        button.st-emotion-cache-7ym5gk.ef3psqc12{
           
            height: 1px;
        }
        .st-gw{
            height: auto;
            width: 300px;
        }
        .st-h9{
            
        }
        </style>
        """,
        unsafe_allow_html=True
    )

        email = st.text_input("Email Address")
        password = st.text_input("Password", type="password")

        login_button_clicked = st.button("LOGIN", key="login_button")
        reset_button_clicked = st.button("Forgot Password", key="reset_button")

        # Create a sign-up button with error handling
        if st.button("Sign Up"):
            if not validate_email(email):
                st.error("Please enter a valid email address.")
            elif not validate_password(password):
                st.error("Password must be at least 6 characters long.")
            else:
                signup(email, password)

        if login_button_clicked:
            if email and password:
                user = authenticate(email, password)
                if user:
                    st.session_state.user = user
                    st.toast("Login successful.")
                    time.sleep(1)
            else:
                st.toast("Please enter both email and password.")
                time.sleep(2)
        if reset_button_clicked:
            send_password_reset_email(email)

     else:
        st.sidebar.image("static/ccs.png", use_column_width=True)

        with open('style.css') as f:
         st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

        # Create tabs
        st.sidebar.header("✅ WELCOME TO LU STUDENT TREND")
        st.sidebar.header("")
        
        tabs = st.sidebar.radio("SELECT TAB:", ["📊 Dashboard", ":chart_with_upwards_trend: Enrollment", ":chart_with_downwards_trend: Dropout", "🎓 Graduate", "🆕 Manage Data", "👤 Account"])


        data = pd.read_csv('data.csv')
        data = data.dropna()

        if tabs == "📊 Dashboard":
                
                st.markdown("<h1 style='color: #E97451;'>📊 Student Trend Dashboard</h1>", unsafe_allow_html=True)
                st.markdown("")   

                fl = st.file_uploader(" UPLOAD A FILE", type=["csv"])

                if fl is not None:
                    # Read the contents of the uploaded file as bytes
                    file_contents = fl.getvalue()

                    # Decode bytes to string assuming it's a CSV file (change encoding if necessary)
                    stringio = io.StringIO(file_contents.decode("ISO-8859-1"))

                    # Use Pandas to read the string as a CSV
                    df = pd.read_csv(stringio)
                    st.write(df)  # Display the DataFrame in Streamlit
                else:
                    df = pd.read_csv("v_data.csv", encoding="ISO-8859-1")

                col1, col2 = st.columns((2))

                st.header("⭐ Choose your filter: ")
                # Create for Program
                program = st.multiselect("Select Program:", df["Program Name"].unique())
                if not program:
                    df2 = df.copy()
                else:
                    df2 = df[df["Program Name"].isin(program)]

                # Create for Year
                year = st.multiselect("Select School Year:", df2["School Year"].unique())
                if not year:
                    df3 = df2.copy()
                else:
                    df3 = df2[df2["School Year"].isin(year)]

                # Filter the data based on Region, State and City

                if not program:
                    filtered_df = df
                elif not year:
                    filtered_df = df[df["Program Name"].isin(program)]
                elif not program:
                    filtered_df = df[df["School Year"].isin(year)]
                elif year:
                    filtered_df = df3[df["School Year"].isin(year)]
                elif program:
                    filtered_df = df3[df["Program Name"].isin(program)]
                elif program and year:
                    filtered_df = df3[df["Program Name"].isin(program) & df3["School Year"].isin(year)]
                else:
                    filtered_df = df3[df3["Program Name"].isin(program) & df3["School Year"].isin(year)]

                program_df = filtered_df.groupby(by = ["Program Name"], as_index = False)["Number of Enrollees"].sum()

                program_df_sorted = program_df.sort_values(by="Number of Enrollees", ascending=False)

                # Select top 5 programs
                top_5_programs = program_df_sorted.head(5)

                # Generate bar chart for top 5 programs
                fig_enrollment = px.bar(
                    top_5_programs,
                    x="Program Name",
                    y="Number of Enrollees",
                    text=['{:,.0f}'.format(x) for x in top_5_programs["Number of Enrollees"]],
                    template="plotly_dark",
                    labels={"Number of Enrollees": "Number of Enrollees"},
                    color="Number of Enrollees",
                    color_continuous_scale=px.colors.sequential.Viridis,
                )

                fig_enrollment.update_traces(textposition='outside')
                fig_enrollment.update_layout(
                    xaxis_title='Program',
                    yaxis_title='Number of Enrollees',
                    showlegend=False,
                    plot_bgcolor='#FFF',
                    paper_bgcolor='#fff',
                    height=500,
                )
                st.plotly_chart(fig_enrollment, use_container_width=True)

                # Expander for Dropout data
                with st.expander("Enrollees Data"):
                    region = filtered_df.groupby(by="Program Name", as_index=False)["Number of Enrollees"].sum()
                    region_top5 = region.nlargest(5, "Number of Enrollees")
                    st.write(region_top5.style.background_gradient(cmap="Oranges"))
                    csv_dropout = region_top5.to_csv(index=False).encode('utf-8')
                    st.download_button('Download Enrollees Data', data=csv_dropout, file_name="Enrollees_Data.csv", mime="text/csv")

                # Get the top 5 programs for dropout
                top5_dropout_programs = filtered_df.groupby(by="Program Name", as_index=False)["Number of Dropout"].sum().nlargest(5, "Number of Dropout")["Program Name"]

                # Filter the DataFrame to include only the top 5 programs
                filtered_df_top5_dropout = filtered_df[filtered_df["Program Name"].isin(top5_dropout_programs)]

                # Generate a pie chart for dropout distribution
                st.markdown("<h2 style=' color: #E97451;'>📉 Dropout</h2>", unsafe_allow_html=True)
                fig_dropout = px.pie(
                    filtered_df_top5_dropout,
                    values="Number of Dropout",
                    names="Program Name",
                    hole=0.5,
                    template="plotly_dark",
                )

                fig_dropout.update_traces(textposition='outside', textinfo='percent+label')
                fig_dropout.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    plot_bgcolor='#FFf',
                    paper_bgcolor='#fff',
                    height=500,
                )

                st.plotly_chart(fig_dropout, use_container_width=True)

                # Expander for Dropout data
                with st.expander("Dropout Data"):
                    region = filtered_df.groupby(by="Program Name", as_index=False)["Number of Dropout"].sum()
                    region_top5 = region.nlargest(5, "Number of Dropout")
                    st.write(region_top5.style.background_gradient(cmap="Oranges"))
                    csv_dropout = region_top5.to_csv(index=False).encode('utf-8')
                    st.download_button('Download Dropout Data', data=csv_dropout, file_name="Dropout_Data.csv", mime="text/csv")

                # Get the top 5 programs for graduates
                top5_programs = filtered_df.groupby(by="Program Name", as_index=False)["Number of Graduates"].sum().nlargest(5, "Number of Graduates")["Program Name"]

                # Filter the DataFrame to include only the top 5 programs
                filtered_df_top5 = filtered_df[filtered_df["Program Name"].isin(top5_programs)]

                # Generate a pie chart for Graduate distribution
                st.markdown("<h2 style=' color: #E97451;'>🎓 Graduates</h2>", unsafe_allow_html=True)
                fig_dropout = px.pie(
                    filtered_df_top5,
                    values="Number of Graduates",
                    names="Program Name",
                    hole=0.5,
                    template="plotly_dark",
                )

                fig_dropout.update_traces(textposition='outside', textinfo='percent+label')
                fig_dropout.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    plot_bgcolor='#FFf',
                    paper_bgcolor='#fff',
                    height=500,
                )

                st.plotly_chart(fig_dropout, use_container_width=True)

                # Expander for Dropout data
                with st.expander("Graduates Data"):
                    region = filtered_df.groupby(by="Program Name", as_index=False)["Number of Graduates"].sum()
                    region_top5 = region.nlargest(5, "Number of Graduates")
                    st.write(region_top5.style.background_gradient(cmap="Oranges"))
                    csv_dropout = region_top5.to_csv(index=False).encode('utf-8')
                    st.download_button('Download Graduates Data', data=csv_dropout, file_name="Graduates_Data.csv", mime="text/csv")



                # Download orginal DataSet
                csv = df.to_csv(index = False).encode('utf-8')
                st.download_button('Download Data', data = csv, file_name = "Data.csv",mime = "text/csv")

        
        with open('stacked_model.pkl', 'rb') as file:
            stacked_model = pickle.load(file)

        if tabs == ":chart_with_upwards_trend: Enrollment":

                st.markdown("<h1 style='color: #E97451;'>📈 Enrollment Prediction</h1>", unsafe_allow_html=True)
                st.markdown("")

                data = pd.read_csv("data.csv", encoding="ISO-8859-1")

                fl = st.file_uploader("UPLOAD A FILE", type=["csv"])

                if fl is not None:
                    # Check if the uploaded file is a CSV
                    if fl.type == 'text/csv':
                        # Read the contents of the uploaded file as bytes
                        file_contents = fl.getvalue()

                        # Decode bytes to string assuming it's a CSV file (change encoding if necessary)
                        stringio = io.StringIO(file_contents.decode("ISO-8859-1"))

                        # Use Pandas to read the string as a CSV
                        uploaded_data = pd.read_csv(stringio)
                        st.write(uploaded_data)  # Display the uploaded DataFrame in Streamlit

                        # Check if the uploaded CSV has the required columns for predictions
                        required_columns = ['School Year', 'Program ID', 'Number of Enrollees']  # Adjust column names as needed
                        if all(col in uploaded_data.columns for col in required_columns):
                            # Check column data types for numeric columns
                            numeric_columns = ['School Year', 'Program ID', 'Number of Enrollees']  # Adjust numeric column names
                            correct_data_types = all(uploaded_data[col].dtype in ['int64', 'float64'] for col in numeric_columns)
                            
                            if correct_data_types:
                                data = uploaded_data  # Update 'data' if columns and data types match the required structure
                            else:
                                st.warning("⚠️ Please upload a CSV file with numeric columns for predictions.")
                        else:
                            st.warning("⚠️ Please upload a valid file format")



                st.sidebar.header("⭐ Predict Enrollees")

                with st.expander("VIEW DATA"):
                    # Replace commas in every column
                    data_no_commas = data.applymap(lambda x: str(x).replace(',', ''))

                    # Display the DataFrame
                    st.write(data_no_commas, format="csv", index=False)

                    # Download button for the original CSV data
                    csv_data = data.to_csv(index=False).encode('utf-8')
                    st.download_button(label="Download CSV", data=csv_data, file_name='data.csv', mime='text/csv')

                    # Button to trigger retraining for the main model
                    if st.button('Retrain Data'):
                        stacked_model = retrain_model(data, stacked_model_filename)
                        if stacked_model:
                            st.write('Data retrained successfully!')
                        else:
                            st.write('Failed to retrain the data. Model file not found!')
                    

                # Get user input for prediction
                sy_input = st.sidebar.number_input("Enter the year: ", step=1)
                id_input = st.sidebar.selectbox("Select Program ID: ", data['Program ID'].unique())

                # Calculate noisy prediction
                noise_scale = 2 + 0.1 * (sy_input - data['School Year'].max())

                # Ensure noise_scale is non-negative
                if noise_scale < 0:
                    noise_scale = 0
                    
                noise = np.random.normal(loc=0, scale=noise_scale)
                

                fig = go.Figure()
                prediction_text = ""
                original_text = ""
                show_recommendation_button = False


                if sy_input != 0 and id_input != 0:
                    user_data = [[sy_input, id_input]]
                    predictions = stacked_model.predict(user_data)
                    noisy_prediction = predictions[0] + noise


                    # Assuming filtered_data contains only the data for the selected Program ID
                    filtered_data = data[data['Program ID'] == id_input]

                    if len(filtered_data) > 0:
                        # Display original data
                        fig.add_trace(go.Scatter(
                            x=filtered_data['School Year'],
                            y=filtered_data['Number of Enrollees'],
                            mode='markers',
                            name='Original Data',
                            marker=dict(color='#FF5733', size=12, line=dict(color='#000000', width=0.5)),
                            opacity=0.8,
                            showlegend=True
                        ))

                        original_value = None  # Initialize original_value here

                        # Check if original value exists for the input year
                        if sy_input in filtered_data['School Year'].values:
                            original_value = filtered_data[filtered_data['School Year'] == sy_input]
                            fig.add_trace(go.Scatter(
                                x=[sy_input],
                                y=[original_value['Number of Enrollees'].values[0]],
                                mode='markers',
                                name='Original Value',
                                marker=dict(color='#C70039', size=16, symbol='diamond', line=dict(color='#000000', width=1.5)),
                            ))
                        else:
                            original_value = None



                        # Predict and display values for 2023 to user-input year
                        if sy_input >= 2023:
                            for year in range(2023, sy_input + 1):
                                noisy_prediction = stacked_model.predict([[year, id_input]])[0]
                                noise_scale = 150 + 0.1 * (year - data['School Year'].max())
                                noise = np.random.normal(loc=0, scale=noise_scale)
                                noisy_prediction += noise
                                noisy_prediction = round(noisy_prediction)

                                # Check if there's an original value for the current year
                                original_value_available = (year in filtered_data['School Year'].values)

                                # Add scatter plot for predicted value only if no original value exists
                                if not original_value_available:
                                    fig.add_trace(go.Scatter(
                                        x=[year],
                                        y=[noisy_prediction],
                                        mode='markers',
                                        name=f'Predicted Value ({year})',
                                        marker=dict(color='#4CAF50', size=16, symbol='star', line=dict(color='#000000', width=1.5)),
                                    ))

                        # Adding a trend line based on historical data if enough data points are available
                        if len(filtered_data) > 1:
                            x = filtered_data['School Year'].values.reshape(-1, 1)
                            y = filtered_data['Number of Enrollees'].values
                            model = LinearRegression().fit(x, y)
                            trend_line = model.predict(x)
                            fig.add_trace(go.Scatter(
                                x=filtered_data['School Year'],
                                y=trend_line,
                                mode='lines',
                                name='Trend Line',
                                line=dict(color='#00FFFF', width=3),
                            ))

                        fig.update_layout(
                            title_text='Enrollees Prediction',
                            title_font=dict(size=28, family='Arial, sans-serif', color='#333333'),
                            title_x=0.33,  # Centers the title horizontally
                            title_y=0.95,  # Adjusts the vertical position of the title
                            xaxis=dict(title='School Year', tickfont=dict(size=14, color='#333333')),
                            yaxis=dict(title='Number of Enrollees', tickfont=dict(size=14, color='#333333')),
                            showlegend=True,
                            legend=dict(
                                x=0,
                                y=1,
                                traceorder="normal",
                                font=dict(family="Arial, sans-serif", size=14, color="#333333"),
                                bgcolor="#f7f7f7",
                                bordercolor="#333333",
                                borderwidth=1
                            ),
                            hovermode='closest',
                            plot_bgcolor='#f0f0f0',  # Background color of the plot
                            paper_bgcolor='#ffffff',  # Background color of the paper/plot area
                            width=800,  # Adjust the width of the plot
                            height=600,  # Adjust the height of the plot
                            margin=dict(l=80, r=80, t=100, b=80),  # Adjust margins for better display
                            transition={'duration': 1000}  # Add smooth transition/animation
                        )
                        prediction_text = f"Predicted Value: {round(noisy_prediction)}"
                        if original_value is not None:
                            original_text = f"Original Value: {int(original_value['Number of Enrollees'].values[0])}"

                        styled_prediction_text = f"<div style='padding: 20px; border-radius: 50px; margin-bottom: 30px; background: linear-gradient(135deg, #50C878, #458B74, #006400); color: #FFF; text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.5); box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), 0 15px 15px rgba(0, 0, 0, 0.3); animation: rotate-scale 3s ease-in-out infinite, neon-glow 2s linear infinite;'><font size='+6'>{prediction_text}</font></div>"
                        styled_original_text = f"<div style='padding: 20px; border-radius: 50px; margin-bottom: 30px; color: #006400; animation: rotate-scale 3s ease-in-out infinite, neon-glow 2s linear infinite;'><font size='+6'>{original_text}</font></div>"

                        st.markdown(styled_prediction_text, unsafe_allow_html=True)
                        st.markdown(styled_original_text, unsafe_allow_html=True)

                        st.plotly_chart(fig)


                        # Checking for significant growth or decrease based on predictions compared to the latest data
                        if original_value is None:
                            latest_value = filtered_data['Number of Enrollees'].iloc[-1]  # Fetching the latest value from the data

                            # Rounding off the values to whole numbers
                            latest_value = round(latest_value)
                            noisy_prediction = max(round(noisy_prediction), 0)

                            # Checking if the predicted value indicates growth compared to the latest data
                            if noisy_prediction > latest_value:
                                growth_amount = noisy_prediction - latest_value
                                growth_report = f"The predicted Enrollees increased by {int(growth_amount)} compared to the latest data."
                                show_growth_recommendation = True
                            else:
                                growth_report = "The predicted Enrollees decreased compared to the latest data."
                                show_growth_recommendation = False

                            # Checking if the predicted value indicates decrease compared to the latest data
                            if noisy_prediction < latest_value:
                                decrease_amount = latest_value - noisy_prediction
                                decrease_report = f"The predicted Enrollees decreased by {int(decrease_amount)} compared to the latest data."
                                show_decrease_recommendation = True
                            else:
                                decrease_report = "The predicted Enrollees increased compared to the latest data."
                                show_decrease_recommendation = False                       

                            # Checking for growth to show recommendation button
                            if show_growth_recommendation:
                                col1, col2, col3 = st.columns([1, 6, 1])
                                with col2:
                                        st.markdown("""
                                            <div style='color: white; font-family: Arial, sans-serif;'>
                                                <div style='margin-bottom: 10px;'>
                                                    <div style='background-color: #2E8B57; padding: 20px; border-radius: 30px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);'>
                                                        <p>{growth_report}</p>
                                                    </div>
                                                </div>
                                            </div>
                                        """.format(growth_report=growth_report), unsafe_allow_html=True)

                                        st.markdown("""
                                            <div style='color: #fff; font-family: Arial, sans-serif;'>
                                                <div style='background-color: #2E8B57; padding: 30px; border-radius: 30px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);'>
                                                    <h2 style='margin-bottom: 20px;'>Recommendation for Handling Increase in Enrollment:</h2>
                                                    <ul style='list-style-type: none; padding-left: 0;'>
                                                        <li style='margin-bottom: 10px;'>✤ Assess and expand infrastructure to accommodate the increased number of students.</li>
                                                        <li style='margin-bottom: 10px;'>✤ Develop additional educational programs and resources to meet the growing demand.</li>
                                                        <li style='margin-bottom: 10px;'>✤ Recruit and train additional faculty/staff to maintain quality education standards.</li>
                                                        <li style='margin-bottom: 10px;'>✤ Enhance support services for students to ensure a smooth transition and adaptation.</li>
                                                        <li style='margin-bottom: 10px;'>✤ Foster community engagement and partnerships to support the growing student population.</li>
                                                    </ul>
                                                </div>
                                            </div>
                                        """, unsafe_allow_html=True)


                            # Checking for decrease to show recommendation button
                            if show_decrease_recommendation:
                                col1, col2, col3 = st.columns([1, 6, 1])
                                with col2:
                                        st.markdown("""
                                            <div style='color: white; font-family: Arial, sans-serif;'>
                                                <div style='margin-bottom: 10px;'>
                                                    <div style='background-color: #2E8B57; padding: 20px; border-radius: 30px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);'>
                                                        <p>{decrease_report}</p>
                                                    </div>
                                                </div>
                                            </div>
                                        """.format(decrease_report=decrease_report), unsafe_allow_html=True)

                                        st.markdown("""
                                            <div style='color: white;'>
                                            <div style='color: #fff; font-family: Arial, sans-serif;'>
                                                <div style='background-color: #2E8B57; padding: 30px; border-radius: 30px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);'>
                                                    <h2 style='margin-bottom: 20px;'>Recommendation for Handling Decrease in Enrollment:</h2>
                                                    <ul style='list-style-type: none; padding-left: 0;'>
                                                        <li style='margin-bottom: 10px;'>✤ Analyze the causes behind the decrease and address potential issues.</li>
                                                        <li style='margin-bottom: 10px;'>✤ Implement retention strategies to prevent further decline in enrollment.</li>
                                                        <li style='margin-bottom: 10px;'>✤ Adjust academic programs to better meet the needs of the reduced student body.</li>
                                                        <li style='margin-bottom: 10px;'>✤ Focus on marketing efforts to attract prospective students and boost enrollment.</li>
                                                        <li style='margin-bottom: 10px;'>✤ Collaborate with community stakeholders to understand and address enrollment challenges.</li>
                                                    </ul>
                                                </div>
                                            </div>
                                        """, unsafe_allow_html=True)



                


        with open('stacked_model_drop.pkl', 'rb') as file:
            stacked_model = pickle.load(file)

            if tabs == ":chart_with_downwards_trend: Dropout":

                st.markdown("<h1 style='color: #E97451;'>📉 Dropout Prediction</h1>", unsafe_allow_html=True)
                st.markdown("")   

                data = pd.read_csv("data.csv", encoding="ISO-8859-1")

                fl = st.file_uploader("UPLOAD A FILE", type=["csv"])

                if fl is not None:
                    # Check if the uploaded file is a CSV
                    if fl.type == 'text/csv':
                        # Read the contents of the uploaded file as bytes
                        file_contents = fl.getvalue()

                        # Decode bytes to string assuming it's a CSV file (change encoding if necessary)
                        stringio = io.StringIO(file_contents.decode("ISO-8859-1"))

                        # Use Pandas to read the string as a CSV
                        uploaded_data = pd.read_csv(stringio)
                        st.write(uploaded_data)  # Display the uploaded DataFrame in Streamlit

                        # Check if the uploaded CSV has the required columns for predictions
                        required_columns = ['School Year', 'Program ID', 'Number of Enrollees']  # Adjust column names as needed
                        if all(col in uploaded_data.columns for col in required_columns):
                            # Check column data types for numeric columns
                            numeric_columns = ['School Year', 'Program ID', 'Number of Enrollees']  # Adjust numeric column names
                            correct_data_types = all(uploaded_data[col].dtype in ['int64', 'float64'] for col in numeric_columns)
                            
                            if correct_data_types:
                                data = uploaded_data  # Update 'data' if columns and data types match the required structure
                            else:
                                st.warning("⚠️ Please upload a CSV file with numeric columns for predictions.")
                        else:
                            st.warning("⚠️ Please upload a valid file format")

                st.sidebar.header("⭐ Predict Dropout")

                with st.expander("VIEW DATA"):
                    # Replace commas in every column
                    data_no_commas = data.applymap(lambda x: str(x).replace(',', ''))

                    # Display the DataFrame
                    st.write(data_no_commas, format="csv", index=False)

                    # Download button for the original CSV data
                    csv_data = data.to_csv(index=False).encode('utf-8')
                    st.download_button(label="Download CSV", data=csv_data, file_name='data.csv', mime='text/csv')

                    # Button to trigger retraining for the dropout model
                    if st.button('Retrain Data'):
                        dropout_model = retrain_dropout_model(data, dropout_model_filename)
                        if dropout_model:
                            st.write('Data retrained successfully!')
                        else:
                            st.write('Failed to retrain the data. Model file not found!')


                # Get user input for prediction
                sy_input = st.sidebar.number_input("Enter the year: ", step=1)
                id_input = st.sidebar.selectbox("Select Program ID: ", data['Program ID'].unique())

                fig = go.Figure()
                prediction_text = ""
                original_text = ""
                show_recommendation_button = False

                if sy_input != 0 and id_input != 0:
                    user_data = [[sy_input, id_input]]
                    predictions = stacked_model.predict(user_data)

                    # Calculate noisy prediction
                    noise_scale = 2 + 0.1 * (sy_input - data['School Year'].max())
                    noise = np.random.normal(loc=0, scale=noise_scale)
                    noisy_prediction = predictions[0] + noise

                    # Ensure noisy prediction is non-negative
                    noisy_prediction = max(noisy_prediction, 0)

                    # Assuming filtered_data contains only the data for the selected Program ID
                    filtered_data = data[data['Program ID'] == id_input]

                    if len(filtered_data) > 0:
                        # Display original data
                        fig.add_trace(go.Scatter(
                            x=filtered_data['School Year'],
                            y=filtered_data['Number of Dropout'],
                            mode='markers',
                            name='Original Data',
                            marker=dict(color='#FF5733', size=12, line=dict(color='#000000', width=0.5)),
                            opacity=0.8,
                            showlegend=True
                        ))

                        original_value = None  # Initialize original_value here

                        # Check if original value exists for the input year
                        if sy_input in filtered_data['School Year'].values:
                            original_value = filtered_data[filtered_data['School Year'] == sy_input]
                            fig.add_trace(go.Scatter(
                                x=[sy_input],
                                y=[original_value['Number of Dropout'].values[0]],
                                mode='markers',
                                name='Original Value',
                                marker=dict(color='#C70039', size=16, symbol='diamond', line=dict(color='#000000', width=1.5)),
                            ))
                        else:
                            original_value = None

                        # Predict and display values for 2023 to user-input year
                        if sy_input >= 2023:
                            for year in range(2023, sy_input + 1):
                                noisy_prediction = stacked_model.predict([[year, id_input]])[0]
                                noise_scale = 150 + 0.1 * (year - data['School Year'].max())
                                noise = np.random.normal(loc=0, scale=noise_scale)
                                noisy_prediction += noise
                                noisy_prediction = round(max(noisy_prediction, 0))  # Ensure it's non-negative

                                # Check if there's an original value for the current year
                                original_value_available = (year in filtered_data['School Year'].values)

                                # Add scatter plot for predicted value only if no original value exists
                                if not original_value_available:
                                    fig.add_trace(go.Scatter(
                                        x=[year],
                                        y=[noisy_prediction],
                                        mode='markers',
                                        name=f'Predicted Value ({year})',
                                        marker=dict(color='#4CAF50', size=16, symbol='star', line=dict(color='#000000', width=1.5)),
                                    ))

                        # Adding a trend line based on historical data if enough data points are available
                        if len(filtered_data) > 1:
                            x = filtered_data['School Year'].values.reshape(-1, 1)
                            y = filtered_data['Number of Dropout'].values
                            model = LinearRegression().fit(x, y)
                            trend_line = model.predict(x)
                            fig.add_trace(go.Scatter(
                                x=filtered_data['School Year'],
                                y=trend_line,
                                mode='lines',
                                name='Trend Line',
                                line=dict(color='#00FFFF', width=3),
                            ))

                        fig.update_layout(
                            title_text='Dropout Prediction',
                            title_font=dict(size=28, family='Arial, sans-serif', color='#333333'),
                            title_x=0.33,  # Centers the title horizontally
                            title_y=0.95,  # Adjusts the vertical position of the title
                            xaxis=dict(title='School Year', tickfont=dict(size=14, color='#333333')),
                            yaxis=dict(title='Number of Dropout', tickfont=dict(size=14, color='#333333')),
                            showlegend=True,
                            legend=dict(
                                x=0,
                                y=1,
                                traceorder="normal",
                                font=dict(family="Arial, sans-serif", size=14, color="#333333"),
                                bgcolor="#f7f7f7",
                                bordercolor="#333333",
                                borderwidth=1
                            ),
                            hovermode='closest',
                            plot_bgcolor='#f0f0f0',  # Background color of the plot
                            paper_bgcolor='#ffffff',  # Background color of the paper/plot area
                            width=800,  # Adjust the width of the plot
                            height=600,  # Adjust the height of the plot
                            margin=dict(l=80, r=80, t=100, b=80),  # Adjust margins for better display
                            transition={'duration': 1000}  # Add smooth transition/animation
                        )
                        prediction_text = f"Predicted Value: {round(noisy_prediction)}"
                        if original_value is not None:
                            original_text = f"Original Value: {int(original_value['Number of Dropout'].values[0])}"

                        styled_prediction_text = f"<div style='padding: 20px; border-radius: 50px; margin-bottom: 30px; background: linear-gradient(135deg, #50C878, #458B74, #006400); color: #FFF; text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.5); box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), 0 15px 15px rgba(0, 0, 0, 0.3); animation: rotate-scale 3s ease-in-out infinite, neon-glow 2s linear infinite;'><font size='+6'>{prediction_text}</font></div>"
                        styled_original_text = f"<div style='padding: 20px; border-radius: 50px; margin-bottom: 30px; color: #006400; animation: rotate-scale 3s ease-in-out infinite, neon-glow 2s linear infinite;'><font size='+6'>{original_text}</font></div>"

                        st.markdown(styled_prediction_text, unsafe_allow_html=True)
                        st.markdown(styled_original_text, unsafe_allow_html=True)

                        st.plotly_chart(fig)

                        # Checking for significant growth or decrease based on predictions compared to the latest data
                        if original_value is None:
                            latest_value = filtered_data['Number of Dropout'].iloc[-1]  # Fetching the latest value from the data

                            # Rounding off the values to whole numbers
                            latest_value = round(latest_value)
                            noisy_prediction = max(round(noisy_prediction), 0)

                            # Checking if the predicted value indicates growth compared to the latest data
                            if noisy_prediction > latest_value:
                                growth_amount = noisy_prediction - latest_value
                                growth_report = f"The predicted Dropouts increased by {int(growth_amount)} compared to the latest data."
                                show_growth_recommendation = True
                            else:
                                growth_report = "The predicted Dropouts decreased compared to the latest data."
                                show_growth_recommendation = False

                            # Checking if the predicted value indicates decrease compared to the latest data
                            if noisy_prediction < latest_value:
                                decrease_amount = latest_value - noisy_prediction
                                decrease_report = f"The predicted Dropouts decreased by {int(decrease_amount)} compared to the latest data."
                                show_decrease_recommendation = True
                            else:
                                decrease_report = "The predicted Dropouts increased compared to the latest data."
                                show_decrease_recommendation = False                       

                                # Checking for growth to show recommendation button
                                if show_growth_recommendation:
                                    col1, col2, col3 = st.columns([1, 6, 1])
                                    with col2:
                                            st.markdown("""
                                                <div style='color: white; font-family: Arial, sans-serif;'>
                                                    <div style='margin-bottom: 10px;'>
                                                        <div style='background-color: #2E8B57; padding: 20px; border-radius: 30px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);'>
                                                            <p>{growth_report}</p>
                                                        </div>
                                                    </div>
                                                </div>
                                            """.format(growth_report=growth_report), unsafe_allow_html=True)
                                            st.markdown("""
                                            <div style='color: #fff; font-family: Arial, sans-serif;'>
                                                <div style='background-color: #2E8B57; padding: 30px; border-radius: 30px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);'>
                                                    <h2 style='margin-bottom: 20px;'>Recommendation for Handling Increase in Dropout Rate:</h2>
                                                    <ul style='list-style-type: none; padding-left: 0;'>
                                                            <li style='margin-bottom: 10px;'>✤ Implement additional support programs, such as mentoring initiatives, tutoring services, and community outreach, to address potential increases in dropout rates.</li>
                                                            <li style='margin-bottom: 10px;'>✤ Focus on early intervention strategies and personalized assistance for at-risk students.</li>
                                                            <li style='margin-bottom: 10px;'>✤ Strengthen communication channels between educators, students, and parents/guardians.</li>
                                                            <li style='margin-bottom: 10px;'>✤ Conduct regular assessments to identify struggling students and provide targeted support.</li>
                                                            <li style='margin-bottom: 10px;'>✤ Collaborate with local organizations or agencies to create comprehensive support networks.</li>
                                                        </ul>
                                                    </div>
                                                </div>
                                            """, unsafe_allow_html=True)


                                # Checking for decrease to show recommendation button
                                if show_decrease_recommendation:
                                    col1, col2, col3 = st.columns([1, 6, 1])
                                    with col2:
                                            st.markdown("""
                                                <div style='color: white; font-family: Arial, sans-serif;'>
                                                    <div style='margin-bottom: 10px;'>
                                                        <div style='background-color: #2E8B57; padding: 20px; border-radius: 30px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);'>
                                                            <p>{decrease_report}</p>
                                                        </div>
                                                    </div>
                                                </div>
                                            """.format(decrease_report=decrease_report), unsafe_allow_html=True)
                                            st.markdown("""
                                            <div style='color: #fff; font-family: Arial, sans-serif;'>
                                                <div style='background-color: #2E8B57; padding: 30px; border-radius: 30px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);'>
                                                        <h2 style='margin-bottom: 20px;'>Recommendation for Decrease in Dropout Rate:</h2>
                                                        <p><strong>Consider maintaining and reinforcing successful programs that contribute to the decrease in dropout rates. Additionally:</strong></p>
                                                        <ul style='list-style-type: none; padding-left: 0;'>
                                                            <li style='margin-bottom: 10px;'>✤ Continue to provide academic and emotional support to students.</li>
                                                            <li style='margin-bottom: 10px;'>✤ Evaluate and expand upon strategies that contributed to the decline.</li>
                                                            <li style='margin-bottom: 10px;'>✤ Encourage community involvement and engagement in educational initiatives.</li>
                                                            <li style='margin-bottom: 10px;'>✤ Monitor trends and adjust interventions based on ongoing assessment and feedback.</li>
                                                        </ul>
                                                    </div>
                                                </div>
                                            """, unsafe_allow_html=True)



                    

        with open('stacked_model_grad.pkl', 'rb') as file:
            stacked_model = pickle.load(file)

            if tabs == "🎓 Graduate":

                    st.markdown("<h1 style='color: #E97451;'>🎓 Graduate Prediction</h1>", unsafe_allow_html=True)
                    st.markdown("")

                    data = pd.read_csv("data.csv", encoding="ISO-8859-1")
                    fl = st.file_uploader("UPLOAD A FILE", type=["csv"])

                    if fl is not None:
                        # Check if the uploaded file is a CSV
                        if fl.type == 'text/csv':
                            # Read the contents of the uploaded file as bytes
                            file_contents = fl.getvalue()

                            # Decode bytes to string assuming it's a CSV file (change encoding if necessary)
                            stringio = io.StringIO(file_contents.decode("ISO-8859-1"))

                            # Use Pandas to read the string as a CSV
                            uploaded_data = pd.read_csv(stringio)
                            st.write(uploaded_data)  # Display the uploaded DataFrame in Streamlit

                            # Check if the uploaded CSV has the required columns for predictions
                            required_columns = ['School Year', 'Program ID', 'Number of Enrollees']  # Adjust column names as needed
                            if all(col in uploaded_data.columns for col in required_columns):
                                # Check column data types for numeric columns
                                numeric_columns = ['School Year', 'Program ID', 'Number of Enrollees']  # Adjust numeric column names
                                correct_data_types = all(uploaded_data[col].dtype in ['int64', 'float64'] for col in numeric_columns)
                                
                                if correct_data_types:
                                    data = uploaded_data  # Update 'data' if columns and data types match the required structure
                                else:
                                    st.warning("⚠️ Please upload a CSV file with numeric columns for predictions.")
                            else:
                                st.warning("⚠️ Please upload a valid file format")

                    st.sidebar.header("⭐ Predict Graduate")

                    with st.expander("VIEW DATA"):
                        # Replace commas in every column
                        data_no_commas = data.applymap(lambda x: str(x).replace(',', ''))

                        # Display the DataFrame
                        st.write(data_no_commas, format="csv", index=False)

                        # Download button for the original CSV data
                        csv_data = data.to_csv(index=False).encode('utf-8')
                        st.download_button(label="Download CSV", data=csv_data, file_name='data.csv', mime='text/csv')

                        # Button to trigger retraining for the graduation model
                        if st.button('Retrain Data'):
                            grad_model = retrain_grad_model(data, grad_model_filename)
                            if grad_model:
                                st.write('Data retrained successfully!')
                            else:
                                st.write('Failed to retrain the data. Model file not found!')
        
                    sy_input = st.sidebar.number_input("Enter the year: ", step=1)
                    id_input = st.sidebar.selectbox("Select Program ID: ", data['Program ID'].unique())
                    en_input = st.sidebar.number_input("Enter no. of Enrollees: ", step=1)

                    fig = go.Figure()
                    prediction_text = ""
                    original_text = ""
                    show_recommendation_button = False

                    if sy_input != 0 and id_input != 0:
                        user_data = [[sy_input, id_input, en_input]]
                        predictions = stacked_model.predict(user_data)

                        filtered_data = data[data['Program ID'] == id_input]

                        fig.add_trace(go.Scatter(
                            x=filtered_data['School Year'],
                            y=filtered_data['Number of Graduates'],
                            mode='markers',
                            name='Original Data',
                            marker=dict(color='#FF5733', size=12, line=dict(color='#000000', width=0.5)),
                            opacity=0.8,
                            showlegend=True
                        ))

                        original_value = None

                        if sy_input in filtered_data['School Year'].values:
                            original_value = filtered_data[filtered_data['School Year'] == sy_input]
                            fig.add_trace(go.Scatter(
                                x=[sy_input],
                                y=[original_value['Number of Graduates'].values[0]],
                                mode='markers',
                                name='Original Value',
                                marker=dict(color='#C70039', size=16, symbol='diamond', line=dict(color='#000000', width=1.5)),
                            ))

                        if original_value is None:  # Only add a star for years without original data
                            fig.add_trace(go.Scatter(
                                x=[sy_input],
                                y=[predictions[0]],
                                mode='markers',
                                name='Predicted Value',
                                marker=dict(color='#4CAF50', size=16, symbol='star', line=dict(color='#000000', width=1.5)),
                            ))
                        else:  # Add the predicted value as well for years with original data
                            fig.add_trace(go.Scatter(
                                x=[sy_input],
                                y=[predictions[0]],
                                mode='markers',
                                name='Predicted Value',
                                marker=dict(color='#4CAF50', size=16, symbol='circle', line=dict(color='#000000', width=1.5)),
                            ))

                        # Adding a trend line based on historical data if enough data points are available
                        if len(filtered_data) > 1:
                            x = filtered_data['School Year'].values.reshape(-1, 1)
                            y = filtered_data['Number of Graduates'].values
                            model = LinearRegression().fit(x, y)
                            trend_line = model.predict(x)
                            fig.add_trace(go.Scatter(
                                x=filtered_data['School Year'],
                                y=trend_line,
                                mode='lines',
                                name='Trend Line',
                                line=dict(color='#00FFFF', width=3),
                            ))

                        # Predict and display values for years with no original data up to the predicted year
                        earliest_year = filtered_data['School Year'].min()
                        for year in range(earliest_year, sy_input):
                            if year not in filtered_data['School Year'].values:
                                predicted_value = stacked_model.predict([[year, id_input, en_input]])[0]
                                predicted_value = round(predicted_value)

                                fig.add_trace(go.Scatter(
                                    x=[year],
                                    y=[predicted_value],
                                    mode='markers',
                                    name=f'Predicted Value ({year})',
                                    marker=dict(color='#4CAF50', size=16, symbol='star', line=dict(color='#000000', width=1.5)),
                                ))

                        fig.update_layout(
                            title_text='Graduate Prediction',
                            title_font=dict(size=28, family='Arial, sans-serif', color='#333333'),
                            title_x=0.33,  # Centers the title horizontally
                            title_y=0.95,  # Adjusts the vertical position of the title
                            xaxis=dict(title='School Year', tickfont=dict(size=14, color='#333333')),
                            yaxis=dict(title='Number of Graduates', tickfont=dict(size=14, color='#333333')),
                            showlegend=True,
                            legend=dict(
                                x=0,
                                y=1,
                                traceorder="normal",
                                font=dict(family="Arial, sans-serif", size=14, color="#333333"),
                                bgcolor="#f7f7f7",
                                bordercolor="#333333",
                                borderwidth=1
                            ),
                            hovermode='closest',
                            plot_bgcolor='#f0f0f0',  # Background color of the plot
                            paper_bgcolor='#ffffff',  # Background color of the paper/plot area
                            width=800,  # Adjust the width of the plot
                            height=600,  # Adjust the height of the plot
                            margin=dict(l=80, r=80, t=100, b=80),  # Adjust margins for better display
                            transition={'duration': 1000}  # Add smooth transition/animation
                        )
                        prediction_text = f"Predicted Value: {round(predictions[0])}"
                        if original_value is not None:
                            original_text = f"Original Value: {original_value['Number of Graduates'].values[0]}"

                        styled_prediction_text = f"<div style='padding: 20px; border-radius: 50px; margin-bottom: 30px; background: linear-gradient(135deg, #50C878, #458B74, #006400); color: #FFF; text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.5); box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), 0 15px 15px rgba(0, 0, 0, 0.3); animation: rotate-scale 3s ease-in-out infinite, neon-glow 2s linear infinite;'><font size='+6'>{prediction_text}</font></div>"
                        styled_original_text = f"<div style='padding: 20px; border-radius: 50px; margin-bottom: 30px; color: #006400; animation: rotate-scale 3s ease-in-out infinite, neon-glow 2s linear infinite;'><font size='+6'>{original_text}</font></div>"

                        st.markdown(styled_prediction_text, unsafe_allow_html=True)
                        st.markdown(styled_original_text, unsafe_allow_html=True)

                        # Checking for significant growth or decrease based on predictions compared to the latest data
                        if original_value is None:
                            latest_value = filtered_data['Number of Graduates'].iloc[-1]  # Fetching the latest value from the data

                            # Rounding off the values to whole numbers
                            latest_value = round(latest_value)
                            predictions[0] = max(round(predictions[0]), 0)

                        st.plotly_chart(fig)

                        # Checking if the predicted value indicates growth compared to the latest data
                        if original_value is None:
                            if predictions[0] > latest_value:
                                growth_amount = predictions[0] - latest_value
                                growth_report = f"The predicted Graduates increased by {int(growth_amount)} compared to the latest data."
                                show_growth_recommendation = True
                            else:
                                growth_report = "The predicted Graduates decreased compared to the latest data."
                                show_growth_recommendation = False


                            # Checking if the predicted value indicates decrease compared to the latest data
                            if predictions[0] < latest_value:
                                decrease_amount = latest_value - predictions[0]
                                decrease_report = f"The predicted Graduates decreased by {int(decrease_amount)} compared to the latest data."
                                show_decrease_recommendation = True
                            else:
                                decrease_report = "The predicted Graduates increased compared to the latest data."
                                show_decrease_recommendation = False                       

                            # Checking for growth to show recommendation button
                            if show_growth_recommendation:
                                col1, col2, col3 = st.columns([1, 6, 1])
                                with col2:

                                        st.markdown("""
                                            <div style='color: white; font-family: Arial, sans-serif;'>
                                                <div style='margin-bottom: 10px;'>
                                                    <div style='background-color: #2E8B57; padding: 20px; border-radius: 30px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);'>
                                                        <p>{growth_report}</p>
                                                    </div>
                                                </div>
                                            </div>
                                        """.format(growth_report=growth_report), unsafe_allow_html=True)
                                        st.markdown("""
                                        <div style='color: #fff; font-family: Arial, sans-serif;'>
                                            <div style='background-color: #2E8B57; padding: 30px; border-radius: 30px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);'>
                                                    <h2 style='margin-bottom: 20px;'>Recommendation for Handling Increase in Graduates:</h2>
                                                    <ul style='list-style-type: none; padding-left: 0;'>
                                                        <li style='margin-bottom: 10px;'>✤ Expand resources and programs to support increased graduation requirements.</li>
                                                        <li style='margin-bottom: 10px;'>✤ Provide additional academic support and counseling services to ensure student success.</li>
                                                        <li style='margin-bottom: 10px;'>✤ Develop career-oriented workshops and mentorship programs to aid in post-graduation plans.</li>
                                                        <li style='margin-bottom: 10px;'>✤ Enhance collaboration with industries for internship opportunities and practical experience.</li>
                                                        <li style='margin-bottom: 10px;'>✤ Evaluate and adapt curriculum to align with evolving job market demands.</li>
                                                    </ul>
                                                </div>
                                            </div>
                                        """, unsafe_allow_html=True)


                            # Checking for decrease to show recommendation button
                            if show_decrease_recommendation:
                                col1, col2, col3 = st.columns([1, 6, 1])
                                with col2:

                                        st.markdown("""
                                            <div style='color: white; font-family: Arial, sans-serif;'>
                                                <div style='margin-bottom: 10px;'>
                                                    <div style='background-color: #2E8B57; padding: 20px; border-radius: 30px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);'>
                                                        <p>{decrease_report}</p>
                                                    </div>
                                                </div>
                                            </div>
                                        """.format(decrease_report=decrease_report), unsafe_allow_html=True)
                                        st.markdown("""
                                            <div style='color: #fff; font-family: Arial, sans-serif;'>
                                                <div style='background-color: #2E8B57; padding: 30px; border-radius: 30px; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);'>
                                                    <h2 style='margin-bottom: 20px;'>Recommendation for Handling Decrease in Graduates:</h2>
                                                    <ul style='list-style-type: none; padding-left: 0;'>
                                                        <li style='margin-bottom: 10px;'>✤ Investigate reasons behind the decrease and address potential issues.</li>
                                                        <li style='margin-bottom: 10px;'>✤ Offer additional support services to improve graduation rates and retention.</li>
                                                        <li style='margin-bottom: 10px;'>✤ Reassess curriculum and educational strategies to better meet student needs.</li>
                                                        <li style='margin-bottom: 10px;'>✤ Strengthen partnerships with industries for job placement and post-graduation opportunities.</li>
                                                        <li style='margin-bottom: 10px;'>✤ Implement proactive measures to identify and support at-risk students.</li>
                                                    </ul>
                                                </div>
                                            </div>
                                        """, unsafe_allow_html=True)



                    


            if tabs == "🆕 Manage Data":
                    

                    st.markdown("<h1 style='color: #E97451;'>🆕 Add New Program Details</h1>", unsafe_allow_html=True)
                    st.markdown("")

                    operation = ""

       
                    operation = st.selectbox("Select Operation", ["Add New Program", "Add New Data", "Edit Data", "Delete"])

                    if operation == "Add New Program":
                        st.subheader("Add New Program")
                        new_program_name = st.text_input("Enter Program Name", key="program_name")
                        new_enrollees = st.number_input("Enter Number of Enrollees", min_value=0, step=1, key="enrollees")
                            
                        if st.button("Add Program"):
                            new_program_id = data['Program ID'].max() + 1
                            new_year = data['School Year'].max()
                            # Assuming default values for dropouts and graduates for a new program
                            new_dropout = 0
                            new_graduates = 0
                            new_row = {'School Year': new_year, 'Program ID': new_program_id,
                                        'Program Name': new_program_name, 'Number of Enrollees': new_enrollees,
                                            'Number of Dropout': new_dropout, 'Number of Graduates': new_graduates}
                            data = data.append(new_row, ignore_index=True)
                            save_data(data)

                                # Check if data was added before showing the success message
                            if len(data) > 0:
                                st.success("New program added successfully!")
                                
                            else:
                                st.warning("No data added. Please fill in the required fields.")



                    if operation == "Add New Data":
                        st.subheader("Add New Data")
                        program_id = st.selectbox("Select Program ID", data['Program ID'].unique())
                        selected_program = data[data['Program ID'] == program_id]['Program Name'].values[0]  # Get the program name for the selected ID
                        new_year = st.number_input("Enter School Year", min_value=int(data['School Year'].max() + 1), step=1, key="new_year")
                        new_enrollees = st.number_input("Enter Number of Enrollees", min_value=0, step=1, key="new_enrollees")
                        new_dropout = st.number_input("Enter Number of Dropout", min_value=0, step=1, key="new_dropout")
                        new_graduates = st.number_input("Enter Number of Graduates", min_value=0, step=1, key="new_graduates")
                        if st.button("Save"):
                            new_row = {'School Year': new_year, 'Program ID': program_id,
                                        'Program Name': selected_program,  # Add the selected program name here
                                        'Number of Enrollees': new_enrollees, 'Number of Dropout': new_dropout,
                                        'Number of Graduates': new_graduates}
                            data = data.append(new_row, ignore_index=True)
                            save_data(data)
                            if len(data) > 0:
                                st.success("New data added successfully!")
                                
                            else:
                                st.warning("No data added. Please fill in the required fields.")
                                        


                    
                    if operation == "Edit Data":
                        st.subheader("Edit Data")
                        
                        # Allow users to select the program and school year to edit
                        program_id_to_edit = st.selectbox("Select Program ID to Edit", data['Program ID'].unique())
                        selected_data = data[data['Program ID'] == program_id_to_edit]
                        
                        if not selected_data.empty:
                            selected_year_to_edit = st.selectbox("Select School Year to Edit", selected_data['School Year'].unique())
                            selected_year_data = selected_data[selected_data['School Year'] == selected_year_to_edit]

                            if not selected_year_data.empty:
                                # Display the existing data for editing
                                st.write("Existing Data:")
                                st.write(selected_year_data)

                                # Allow editing the fields
                                new_enrollees = st.number_input("Enter New Number of Enrollees", min_value=0, step=1, key="edit_enrollees")
                                new_dropout = st.number_input("Enter New Number of Dropout", min_value=0, step=1, key="edit_dropout")
                                new_graduates = st.number_input("Enter New Number of Graduates", min_value=0, step=1, key="edit_graduates")

                                if st.button("Update Data"):
                                    # Update the selected data with the new values
                                    data.loc[(data['Program ID'] == program_id_to_edit) & (data['School Year'] == selected_year_to_edit), 'Number of Enrollees'] = new_enrollees
                                    data.loc[(data['Program ID'] == program_id_to_edit) & (data['School Year'] == selected_year_to_edit), 'Number of Dropout'] = new_dropout
                                    data.loc[(data['Program ID'] == program_id_to_edit) & (data['School Year'] == selected_year_to_edit), 'Number of Graduates'] = new_graduates
                                    save_data(data)
                                    st.success("Data updated successfully!")

                            else:
                                st.warning("No data found for the selected School Year.")
                        else:
                            st.warning("No data found for the selected Program ID.")

                    
                    if operation == "Delete":
                        st.subheader("Delete Data")

                        # Create a drop-down selectbox for Program ID
                        program_id_options = data['Program ID'].unique()
                        selected_program_id = st.selectbox("Select Program ID to Delete", program_id_options)

                        year_to_delete = st.number_input("Enter School Year to Delete", min_value=int(data['School Year'].min()), step=1, key="delete_year")

                        # Display data of the chosen user ID before deletion
                        st.subheader("Data of Chosen User ID Before Deletion")
                        chosen_user_data = data[(data['Program ID'] == selected_program_id) & (data['School Year'] == year_to_delete)]
                        st.write(chosen_user_data)

                        if st.button("Confirm Deletion", key="delete_button"):
                            # Check if data exists for the specified program ID and school year
                            # This line filters the data based on the conditions
                            if ((data['Program ID'] == selected_program_id) & (data['School Year'] == year_to_delete)).any():
                                # Remove data for the specified program ID and school year
                                data = data[~((data['Program ID'] == selected_program_id) & (data['School Year'] == year_to_delete))]
                                # Assuming you have a function like save_data(data) to save the DataFrame
                                save_data(data)
                                st.success("Data deleted successfully!")
                                
                            else:
                                st.warning("No data found for the specified Program ID and School Year.")
           
            if tabs == "👤 Account":

                current_user = get_current_logged_in_user()

                if current_user and 'email' in current_user:
                    if current_user['email'] == "Hesoyam@gmail.com":
                        st.subheader("MANAGE ACCOUNT")
                        try:
                            manage_accounts()
                        except Exception as e:
                            st.error(f"Error while managing accounts: {e}")
                    else:
                        st.write("Please log in with the account registered to REGISTRAR to view the content.")
                else:
                    st.write("Please log in to view the content.")
            

                

        if st.sidebar.button("Logout", on_click=logout):
            logout()  # Call the logout function to clear session state

if __name__ == "__main__":
    main()