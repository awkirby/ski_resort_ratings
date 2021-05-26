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

st.markdown('**Welcome to Ski Resort Manager!**')
st.markdown('A web-app that allows you to explore ski resorts from across the world, '
            'rate them using pre-trained machine learning algorithms, '
            'and explore how changes to key features can improve the customer experience.')
st.markdown('And if your resorts is not listed or you want to make your own resort tycoon style, '
            'that option is also available.')
st.markdown('**How to use**')
st.markdown('1. Select your resort (if available) or create it if it is not')
st.markdown('2. Get a rating')
st.markdown('3. Then have a go at improving it')
st.markdown('**Have fun!**')

# Load stuff
df_resorts = pd.read_csv("data/resorts_data_for_ratings.csv", index_col=0)    # Original Data
df_resorts_ml = pd.read_csv("data/resort_data_classifier_ready.csv", index_col=0)  # Data used in model
model = pickle.load(open('random_forest_model.pkl', 'rb'))                    # Model

st.sidebar.write("Select details of your current resort\n")

# Add selection options for location and resort
continent = st.sidebar.selectbox('Continent:', sorted(df_resorts['Continent'].unique()))

# Create a sorted list of unique countries, then insert a Not Available option
countries = sorted(df_resorts['Country'][df_resorts['Continent'] == continent].unique())
countries.insert(0, "Not Available")

country = st.sidebar.selectbox('Country (if available):', countries)

if country != "Not Available":

    resorts = sorted(df_resorts['Name'][df_resorts['Country'] == country].unique())
    resorts.insert(0, "Not Available")

    resort = st.sidebar.selectbox('Select your resort (if available):', resorts)

else:
    resort = "Not Available"

# Adapt interface based on selection
if resort != "Not Available":
    # Get index
    resort_index = df_resorts[df_resorts['Name'] == resort].index[0]

    # Check if the cost in euros is missing
    if df_resorts.loc[resort_index].any(skipna=False):
        pass_cost = st.number_input("Cost of day pass is missing, please enter:", 0.0, 300.0)
        df_resorts.loc[resort_index, "Cost in Euros"] = pass_cost

    # Display resort information
    left, _ = st.beta_columns(2)
    with left:
        st.table(df_resorts[df_resorts['Name'] == resort].drop(columns=['Star Rating']).T)

    # Place resort information into correct format for classification

    # Some resorts are not in the classifier-ready DataFrame. Check
    try:
        # Assign the feature vector to a variable
        resort_for_prediction = np.asarray(df_resorts_ml.drop(columns=["Star Rating"]).loc[resort_index])

    except KeyError:
        # Get the Dataframe in the format that the model trained in
        df_resorts_prediction = df_resorts.drop(columns=[
            "Name",
            "Max Elevation (m)",
            "Black Piste Percent",
            "Star Rating"
        ])
        df_resorts_prediction = pd.get_dummies(df_resorts_prediction, columns=["Continent", "Country"])

        # Take only the resort that we are predicting
        # Conversion to array gives access to the .reshape() method
        resort_for_prediction = np.asarray(df_resorts_prediction.loc[resort_index])


else:
    # Get a name for the resort
    resort = st.sidebar.text_input("Your resort:", value="<enter resort name>")

    if resort == "<enter resort name>":
        st.sidebar.warning("Please input a name")
        st.stop()

    st.success("\nTell us about " + resort)

    # Get the elevation information
    min_elevation = st.number_input("Base elevation, metres", 0, 4000, key='min')
    max_elevation = st.number_input("Peak elevation, metres", 0, 4000, key='max')
    elevation_change = max_elevation - min_elevation

    # Piste Information
    piste_length = st.number_input("Total length of ski pistes, kilometres", 0.0, 3000.0, 1.0)
    piste_length_blue = st.number_input("Length of blue ski pistes, kilometres", 0.0, piste_length)
    piste_length_red = st.number_input("Length of red ski pistes, kilometres", 0.0, piste_length - piste_length_blue)
    piste_length_black = st.number_input("Length of black ski pistes, kilometres",
                                             piste_length - piste_length_blue - piste_length_red,
                                             piste_length - piste_length_blue - piste_length_red)

    # Ski Lifts
    ski_lifts = st.number_input("Total number of ski lifts", 1, 200)
    # Ski Pass
    pass_cost = st.number_input("Cost of a day pass in Euros", 0.0, 300.0)

    # Construct the feature vector (keep naming consistent with selection from existing options)
    # Get the number of features in the classifier, minus the target feature
    num_features = len(df_resorts_ml.drop(columns=["Star Rating"]).columns)
    resort_for_prediction = np.zeros(num_features)

    # Note some data inputs are dropped for reasons of collinearity
    resort_for_prediction[0] = elevation_change
    resort_for_prediction[1] = min_elevation
    resort_for_prediction[2] = piste_length
    resort_for_prediction[3] = (piste_length_blue / piste_length) * 100  # Piste breakdown is in %s
    resort_for_prediction[4] = (piste_length_red / piste_length) * 100
    resort_for_prediction[5] = ski_lifts
    resort_for_prediction[6] = pass_cost
    # Include the location information, where available
    # Use the index() method to get the location in the feature vector for a specific Continent or Country
    # Use list() function to account for the Index object not having an index() method
    continent_position = list(df_resorts_ml.drop(columns=["Star Rating"]).columns).index("Continent_" + continent)
    resort_for_prediction[continent_position] = 1

    # Check if the resort is in an existing country
    if country is not "Not Available":
        country_position = list(df_resorts_ml.drop(columns=["Star Rating"]).columns).index("Country_" + country)
        resort_for_prediction[country_position] = 1

_, col2, _ = st.beta_columns(3)

with col2:
    if st.button("Get Rating"):
        st.success("The predicted rating\n for your resort is:")
        rating = model.predict(resort_for_prediction.reshape(1, -1))
        st.warning(rating[0].capitalize() + '!')

st.write("Make changes to improve your rating")
make_changes = st.radio("Make Changes?", ["Yes", "No"], index=1)

if make_changes == "No":
    st.warning("No changes requested")
    st.stop()

# Create new resort
updated_resort = np.copy(resort_for_prediction)

# Select feature to change
features = ["Ski Lifts", "Cost", "Pistes", "Elevation Details"]
features_to_update = st.multiselect('Select features to change', features)

if "Ski Lifts" in features_to_update:
    updated_ski_lifts = st.number_input("Updated number of Ski Lifts",
                                    min_value=1, max_value=200, value=int(updated_resort[5]))
    # Update to new value
    updated_resort[5] = updated_ski_lifts

if "Cost" in features_to_update:
    updated_cost = st.number_input("Update cost of ski pass (in Euros)",
                                            min_value=0.0, max_value=300.0, value=updated_resort[6], step=1.00)
    # Update to new value
    updated_resort[6] = updated_cost

if "Pistes" in features_to_update:
    updated_piste_length = st.number_input("Updated piste length (in kilometres)",
                                            min_value=0.0, max_value=3000.0, value=updated_resort[2], step=0.1)
    st.text(f'Current piste breakdown:\n' +
            f'Blue: {updated_resort[3]}%\n' +
            f'Red: {updated_resort[4]}%\n' +
            f'Black: {100 - updated_resort[4] - updated_resort[3]}%')
    updated_piste_length_blue = st.number_input("Length of blue ski pistes, kilometres",
                                                0.0, updated_piste_length,
                                                value=updated_resort[3] * updated_piste_length / 100,
                                                key='update_blue')
    updated_piste_length_red = st.number_input("Length of red ski pistes, kilometres",
                                               0.0, updated_piste_length,
                                               value=updated_resort[4] * updated_piste_length / 100,
                                               key='update_red')
    updated_piste_length_black = st.number_input("Length of black ski pistes, kilometres",
                                                 0.0, updated_piste_length,
                                                 value=updated_piste_length - (updated_piste_length_red +
                                                                               updated_piste_length_blue),
                                                 key='updated_black')
    # Update the values
    updated_resort[2] = updated_piste_length
    updated_resort[3] = (updated_piste_length_blue / updated_piste_length) * 100  # Piste breakdown is in %s
    updated_resort[4] = (updated_piste_length_red / updated_piste_length) * 100

    # Display New Breakdown
    st.text(f'New breakdown:\n' +
            f'Blue: {round(updated_resort[3], 2)}%\n' +
            f'Red: {round(updated_resort[4], 2)}%\n' +
            f'Black: {round((updated_piste_length_black / updated_piste_length) * 100, 2)}%')

if "Elevation Details" in features_to_update:
    # New elevation information
    min_elevation = st.number_input("Base elevation, metres", 0, 4000, value=int(updated_resort[1]), key='min_update')
    max_elevation = st.number_input("Peak elevation, metres", 0, 4000,
                                    value=int(updated_resort[1] + updated_resort[0]),
                                    key='max_update')
    updated_resort[1] = min_elevation
    updated_resort[0] = max_elevation - updated_resort[1]

_, col2, _ = st.beta_columns(3)

with col2:
    if st.button("Get New Rating"):
        st.success("The new rating\n for your resort is:")
        rating = model.predict(updated_resort.reshape(1, -1))
        st.warning(rating[0].capitalize() + '!')

