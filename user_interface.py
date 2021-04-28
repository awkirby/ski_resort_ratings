"""
This interface script allows a user to interact with the pre-trained model of the ski resort classifier.
The user can adjust specific features of a ski resort and the script will return a rating of
"excellent", "good" or "low".
This helps the user determine the best approach to invest in their ski resort, i.e. ski resort tycoon support tool
"""
import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.title('Ski Resort Manager')

st.write("Select details of your current resort\n")

# Load stuff
df_resorts = pd.read_csv("data/resorts_data_for_ratings.csv", index_col=0)    # Original Data
df_resorts_ml = pd.read_csv("data/resort_data_classifier_ready.csv", index_col=0)  # Data used in model
model = pickle.load(open('random_forest_model.pkl', 'rb'))                    # Model

# Add selection options for location and resort
continent = st.selectbox('Continent:', sorted(df_resorts['Continent'].unique()))

country = st.selectbox('Country (if available):',
                       sorted(df_resorts['Country'][df_resorts['Continent'] == continent].unique()))

country_exists = st.radio("Is your country available?", ["Yes", "No"])

if country_exists == "Yes":

    resort = st.selectbox('Select your resort (if available):',
                          sorted(df_resorts['Name'][df_resorts['Country'] == country].unique()))

    resort_exists = st.radio("Is your resort available?", ["Yes", "No"])

else:
    resort_exists = "No"

# Adapt interface based on selection
if resort_exists == 'Yes':
    # Get index
    resort_index = df_resorts[df_resorts['Name'] == resort].index[0]

    # Check if the cost in euros is missing
    if df_resorts.loc[resort_index].any(skipna=False):
        pass_cost = st.slider("Cost of day pass is missing, please enter:", 0.0, 300.0)
        df_resorts.loc[resort_index, "Cost in Euros"] = pass_cost

    # Display resort information (To be improved!)
    st.table(df_resorts[df_resorts['Name'] == resort].drop(columns=['Star Rating']))

else:
    # Get a name for the resort
    resort = st.text_input("Your resort:", value="<enter resort name>")

    st.write("Tell us about ", resort)

    # Get the elevation information
    min_elevation = st.number_input("Base elevation, metres", 0, 4000)
    max_elevation = st.number_input("Peak elevation, metres", 0, 4000)
    elevation_change = max_elevation - min_elevation

    # Piste Information
    piste_length = st.number_input("Total length of ski pistes, kilometres", 0.0, 3000.0)
    piste_length_blue = st.number_input("Length of blue ski pistes, kilometres", 0.0, piste_length)
    piste_length_red = st.number_input("Length of red ski pistes, kilometres", 0.0, piste_length - piste_length_blue)
    piste_length_black = st.number_input("Length of black ski pistes, kilometres",
                                         piste_length - piste_length_blue - piste_length_red,
                                         piste_length - piste_length_blue - piste_length_red)

    # Ski Lifts
    ski_lifts = st.number_input("Total number of ski lifts", 1, 200)
    # Ski Pass
    pass_cost = st.slider("Cost of a day pass in Euros", 0.0, 300.0)

col1, col2, col3 = st.beta_columns(3)

with col2:
    if st.button("Get Rating"):
        if resort_exists == "Yes":
            # Some resorts are not in the classifier-ready DataFrame. Check
            try:
                # Convert to an array in order to allow the .predict method to take in a single input
                rating = model.predict(np.asarray(
                                         df_resorts_ml.drop(columns=["Star Rating"]).loc[resort_index]).reshape(1, -1))
                st.text(rating[0].capitalize() + '!')
            except KeyError:
                # Need to format data (see "supervised_learning.ipynb" for more information)
                resort_for_prediction = df_resorts.drop(columns=[
                                                                "Name",
                                                                "Max Elevation (m)",
                                                                "Black Piste Percent",
                                                                "Star Rating"
                                                                ]).loc[resort_index]
                resort_for_prediction = pd.get_dummies(resort_for_prediction, columns=["Continent", "Country"])
                rating = model.predict(np.asarray(resort_for_prediction).reshape(1, -1))
                st.text(rating[0].capitalize() + '!')

        else:
            st.text("Under Construction")

make_changes = st.radio("Would you like to make changes to the resort?", ["Yes", "No"])



# Get all the features in a dictionary
base_resort = {}

#base_resort['Total Piste Length (km)'] = float(input("Enter the total piste length in km:\n"))
#blue_piste_length = float(input("Enter the total length of blue pistes in km (i.e. 3.3):\n"))
#red_piste_length = float(input("Enter the total length of red pistes in km (i.e. 2.0):\n"))
#black_piste_length = base_resort['Total Piste Length (km)'] - blue_piste_length - red_piste_length

