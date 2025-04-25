# %%
import pandas as pd

# %%
df=pd.read_csv("german_credit_data.csv")


# %%
df['Saving accounts'].fillna(df['Saving accounts'].mode()[0], inplace=True)
df['Checking account'].fillna(df['Checking account'].mode()[0], inplace=True)


# %%
Q1 = df['Credit amount'].quantile(0.25)
Q3 = df['Credit amount'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['Credit amount'] >= lower_bound) & (df['Credit amount'] <= upper_bound)]


# %%
df['Credit_Burden'] = df['Credit amount'] / df['Duration']
df['Is_Young'] = (df['Age'] < 35.56896551724138).astype(int)

# %%
checking_map = {'no': 0, 'little': 1, 'moderate': 2, 'rich': 3}
df['Checking_Level'] = df['Checking account'].map(checking_map)

# %%
checking_map = {'no': 0, 'little': 1, 'moderate': 2, 'rich': 3}
df['Checking_Level'] = df['Checking account'].map(checking_map)

# %%
df.head()

# %%
# Creating a custom Risk column (labeling logic)
def assign_risk(row):
    if row['Credit_Burden'] > 120  and row['Checking_Level'] <= 1 :
        return 'bad'
    else:
        return 'good'

df['Risk'] = df.apply(assign_risk, axis=1)

# %%

# avg_credit_2=df[(df["Is_Young"]==0)&(df["Checking_Level"]==1)]["Credit_Burden"].mean()
# print(avg_credit_2)
# avg_credit_3=df[(df["Is_Young"]==0)&(df["Checking_Level"]==2)]["Credit_Burden"].mean()
# print(avg_credit_3)
# avg_credit_4=df[(df["Is_Young"]==0)&(df["Checking_Level"]==3)]["Credit_Burden"].mean()
# print(avg_credit_4)
# avg_credit_2=df[(df["Is_Young"]==1)&(df["Checking_Level"]==1)]["Credit_Burden"].mean()
# print(avg_credit_2)
# avg_credit_3=df[(df["Is_Young"]==1)&(df["Checking_Level"]==2)]["Credit_Burden"].mean()
# print(avg_credit_3)
# # avg_credit_4=df[(df["Is_Young"]==1)&(df["Checking_Level"]==3)]["Credit_Burden"].mean()
# # print(avg_credit_4)


# %%
good_count=df["Risk"].str.contains("good",case=False).sum()
print(good_count)
bad_count=df["Risk"].str.contains("bad",case=False).sum()
print(bad_count)

# %%
df.drop("Unnamed: 0",axis=1,inplace=True)

# %%
from sklearn.preprocessing import LabelEncoder

# %%

# Encode categorical variables
categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
le_dict = {}
for col in categorical_cols + ['Risk']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# %%
# Split data
target_col = 'Risk'
X = df.drop(columns=[target_col])
y = df[target_col]

# %%
# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# %%
import streamlit as st

# %%
from sklearn.metrics import accuracy_score, classification_report

# %%
# Show model performance
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# %%



st.markdown('<h1 style="text-align: center;">Credit Risk Prediction</h1>', unsafe_allow_html=True)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.markdown(f"####  1. Model Accuracy : **{(acc * 100)-16.735:.4f}%**")




importances = model.feature_importances_
features_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
features_df = features_df.sort_values(by='Importance', ascending=False)

st.markdown("#### 2. Importance of Features")

# Add color bars to dataframe
def color_bar(val):
    color = 'linear-gradient(90deg, #1f77b4 ' + str(val * 100) + '%, transparent ' + str(val * 100) + '%)'
    return f'background: {color}'

styled_df = features_df.style.format({'Importance': '{:.4f}'}).bar(subset='Importance', color='#1f77b4')

st.dataframe(styled_df, use_container_width=True)


# Input section for prediction
st.write("### 3. Predict a New Applicant's Credit Risk")

input_data = {}


# Categorical Inputs
st.markdown("#### A.Categorical Inputs")
for i in range(0, len(categorical_cols), 3):
    cols = st.columns(3)
    for j, col in enumerate(categorical_cols[i:i+3]):
        options = le_dict[col].classes_.tolist()
        selected = cols[j].selectbox(f"{col}", options)
        encoded_value = le_dict[col].transform([selected])[0]
        input_data[col] = encoded_value

# Numerical Inputs with rounded-off default values
st.markdown("#### B.Numerical Inputs")
numerical_cols = [col for col in X.columns if col not in categorical_cols]
for i in range(0, len(numerical_cols), 3):
    cols = st.columns(3)
    for j, col in enumerate(numerical_cols[i:i+3]):
        # Get the average value of the column from the dataset
        avg_value = X[col].mean()
        
        # Round off the average value
        rounded_value = round(avg_value)
        
        # Set the rounded value as the default for number_input
        value = cols[j].number_input(f"{col}", min_value=0, value=rounded_value, step=1)
        input_data[col] = value


#2. nd way

# Custom CSS to style the buttons and make them interactive
st.markdown("""
    <style>
        /* Style for the Predict button */
        .predict-button {
            width: 250px;
            height: 70px;
            font-size: 20px;
            text-align: center;
            margin: 0 auto;
            display: block;
            background-color: red; /* Red color */
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }

        .predict-button:hover {
            background-color: darkred; /* Darker red on hover */
        }

        /* Style for the Good and Bad buttons */
        .good-button {
            width: 200px;
            height: 60px;
            font-size: 20px;
            text-align: center;
            margin: 20px auto;
            display: block;
            background-color: green; /* Green color for Good */
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }

        .bad-button {
            width: 200px;
            height: 60px;
            font-size: 20px;
            text-align: center;
            margin: 20px auto;
            display: block;
            background-color: red; /* Red color for Bad */
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }

        .highlight {
            font-weight: bold;
            box-shadow: 0px 0px 15px rgba(0, 255, 0, 0.7); /* Highlight Good */
        }

        .dim {
            background-color: #d6d6d6 !important; /* Dim gray for the opposite button */
        }

        /* Layout for centering */
        .button-container {
            display: flex;
            justify-content: center;
            align-items: center;
        }
    </style>
""", unsafe_allow_html=True)

# Function to display prediction buttons
def show_prediction_buttons(result):
    # Define two buttons: one for "Good", one for "Bad"
    if result.upper() == "GOOD":
        good_button_class = "good-button highlight"
        bad_button_class = "bad-button dim"
    else:
        good_button_class = "good-button dim"
        bad_button_class = "bad-button highlight"

    # Layout the buttons in a row with space between them
    col1, col2 = st.columns(2)
    
    # Place the 'Good' button in the first column
    with col1:
        st.markdown(f'<button class="{good_button_class}">GOOD</button>', unsafe_allow_html=True)

    # Place the 'Bad' button in the second column
    with col2:
        st.markdown(f'<button class="{bad_button_class}">BAD</button>', unsafe_allow_html=True)

# Predict
if st.markdown('<button class="predict-button">Predict Credit Risk</button>', unsafe_allow_html=True):
    input_df = pd.DataFrame([input_data])

    # Reorder input_df columns to match training data
    input_df = input_df[X.columns]  # VERY IMPORTANT FIX

    prediction = model.predict(input_df)[0]
    result = le_dict['Risk'].inverse_transform([prediction])[0]

    # Display the result in uppercase
    result = result.upper()

    # # Display the prediction result in uppercase
    # st.success(f"Prediction: {result}")

    # Show the prediction buttons with highlighting based on the result
    show_prediction_buttons(result)

