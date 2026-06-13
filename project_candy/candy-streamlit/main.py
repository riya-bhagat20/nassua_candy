import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# # Title
# st.title("Simple Data Plot App 📊")

# # Step 1: Simple Database (DataFrame)
# data = {
#     "Name": ["A", "B", "C", "D", "E"],
#     "Sales": [100, 200, 150, 300, 250]
# }

# df = pd.DataFrame(data)

# # Show Data
# st.subheader("Data Table")
# st.write(df)

# # Step 2: Plot Graph
# st.subheader("Sales Chart")
# st.bar_chart(df.set_index("Name"))

# ---------line plot---------------

# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Title
# st.title("Seaborn Plot in Streamlit 📊")

# # Simple Data (Database)
# data = {
#     "Name": ["A", "B", "C", "D", "E"],
#     "Sales": [100, 200, 150, 300, 250]
# }

# df = pd.DataFrame(data)


# selected_name = st.selectbox("Select Name:", df["Name"])

# # Filter Data
# filtered_df = df[df["Name"] == selected_name]

# # Show Result
# st.write("Filtered Data:")
# st.write(filtered_df)

# # Show Data
# st.write(df)

# # # Seaborn Plot
# # st.subheader("Bar Plot (Seaborn)")

# # fig, ax = plt.subplots()
# # sns.barplot(x="Name", y="Sales", data=df, ax=ax)

# # # Show Plot in Streamlit
# # st.pyplot(fig)

# # ---------------------drop down-------------------------


# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# st.title("Seaborn Interactive Dashboard 🎯")

# # Sample Data
# data = {
#     "Category": ["Electronics", "Clothing", "Grocery", "Electronics", "Clothing", "Grocery"],
#     "Product": ["Mobile", "Shirt", "Rice", "Laptop", "Jeans", "Wheat"],
#     "Sales": [200, 150, 100, 300, 250, 180]
# }

# df = pd.DataFrame(data)

# # Sidebar Dropdown (Slicer)
# selected_category = st.sidebar.selectbox(
#     "Select Category:", df["Category"].unique()
# )

# # Filter Data
# filtered_df = df[df["Category"] == selected_category]

# # Show Data
# st.write("Filtered Data")
# st.write(filtered_df)

# # Seaborn Style
# sns.set_style("darkgrid")

# # Plot
# st.subheader("Sales Chart (Seaborn)")
# fig, ax = plt.subplots()

# sns.barplot(
#     x="Product",
#     y="Sales",
#     data=filtered_df,
#     ax=ax
# )

# # Show in Streamlit
# st.pyplot(fig)




# Page Title
st.title("📊 Care Load Analytics Dashboard")


df = pd.read_csv('Nassau Candy Distributor.csv')

# create check box

Region = st.radio("Select Drink", ['milk', 'water', 'hot'])
st.write({Region})

# create select box

flaour=st.selectbox('choose flevour',['gingur','kaser','masala','rose','chocolate','elaychi'])

# create slicer

sugar=st.slider('sugar leve',0,10,1)
st.write(f'how many teble spoon sugar : {sugar}')

# number input

cups=st.number_input('how many cups tea',min_value=1,max_value=10,step=1)
st.write({cups})

# text input

name=st.text_input('whats your name')
if name:
    st.write(f'hii-----------,{name} !')

dates=st.date_input('how many cups tea')
st.write({dates})


# add button

if st.button('Say hello'):
    st.write('why hello say')
else :
    st.write('goodbye')    
   

option =st.sidebar.selectbox('your city',('bhopal','delhi','indor','balaghat','jablpur'))
st.write({option})   