import os
from random import randrange

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from mylib.Run import run

st.title('Analytics')

"""
# Business Analysis of Restautant's profit
Using People Counter and Machine learning
"""
cost_price_model_path = 'models/cost_price/'
final_price_model_path = 'models/final_price/'
final_rating_model_path = 'models/final_rating/'
final_sold_model_path = 'models/final_sold/'
    
price = joblib.load(cost_price_model_path+ 'price.pkl')

chettinad_mutton_price = joblib.load(final_price_model_path + 'chettinad_mutton_price.pkl')
chicken_curry_price = joblib.load(final_price_model_path + 'chicken_curry_price .pkl')
chicken_nuggets_price = joblib.load(final_price_model_path + 'chicken_nuggets_price.pkl')
dhal_makni_price = joblib.load(final_price_model_path + 'dhal_makni_price.pkl')
mutton_chops_price = joblib.load(final_price_model_path + 'mutton_chops_price.pkl')
paneer_tikka_price = joblib.load(final_price_model_path + 'paneer_tikka_price.pkl')
prawn_fry_price = joblib.load(final_price_model_path + 'prawn_fry_price.pkl')
veg_manchurian_price = joblib.load(final_price_model_path + 'veg_manchurian_price.pkl')

chettinad_mutton_rating = joblib.load(final_rating_model_path + 'chettinad_mutton_rating.pkl')
chicken_curry_rating = joblib.load(final_rating_model_path + 'chicken_curry_rating.pkl')
chicken_nuggets_rating = joblib.load(final_rating_model_path + 'chicken_nuggets_rating.pkl')
dhal_makni_rating = joblib.load(final_rating_model_path + 'dhal_makni_rating.pkl')
mutton_chops_rating = joblib.load(final_rating_model_path + 'mutton_chops_rating.pkl')
paneer_tikka_rating = joblib.load(final_rating_model_path + 'paneer_tikka_rating.pkl')
prawn_fry_rating = joblib.load(final_rating_model_path + 'prawn_fry_rating.pkl')
veg_manchurian_rating = joblib.load(final_rating_model_path + 'veg_manchurian_rating.pkl')

chettinad_mutton_plates = joblib.load(final_sold_model_path + 'chettinad_mutton_plates.pkl')
chicken_curry_plates = joblib.load(final_sold_model_path + 'chicken_curry_plates.pkl')
chicken_nuggets_plates = joblib.load(final_sold_model_path + 'chicken_nuggets_plates.pkl')
dhal_makni_plates = joblib.load(final_sold_model_path + 'dhal_makni_plates.pkl')
mutton_chops_plates = joblib.load(final_sold_model_path + 'mutton_chops_plates.pkl')
paneer_tikka_plates = joblib.load(final_sold_model_path + 'paneer_tikka_plates.pkl')
prawn_fry_plates = joblib.load(final_sold_model_path + 'prawn_fry_plates.pkl')
veg_manchurian_plates = joblib.load(final_sold_model_path + 'veg_manchurian_plates.pkl')

if st.button('Start People Counting?'):
    """
    [INFO] Starting the live stream..
    """
    run()
        


def table():

    week = 1
    month = 1
    year = randrange(5)
    date = f'{month}/2021'

    #while week <= 4 and month <= 12:
    for (i, (k, v)) in enumerate(run.info2):
        if v > 100:
            Price_data = price.predict([[year, month, week]]) + 5000
        elif v > 50:
            Price_data = price.predict([[year, month, week]]) + 1000
        elif v > 20:
            Price_data = price.predict([[year, month, week]]) + 600
        



    df = pd.DataFrame({
'Date': f'{date}',
'Week': f'{week}',
'Price': Price_data,
'chettinad_mutton_plates' : chettinad_mutton_plates.predict([[year, month, week]]),
'chettinad_mutton_rating' : chettinad_mutton_rating.predict([[year, month, week]]),
'chettinad_mutton_price' : chettinad_mutton_price.predict([[year, month, week]]),
'Discount value' : 'prediction'
}, index=[0])

    st.table(df)#,width = 10000)

table()

def main():
    html_temp = """<div class='tableauPlaceholder' id='viz1612960191058' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Re&#47;Restaurant_16128584650850&#47;TotalPriceYearwise&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Restaurant_16128584650850&#47;TotalPriceYearwise' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Re&#47;Restaurant_16128584650850&#47;TotalPriceYearwise&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1612960191058');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"""
    components.html(html_temp, height=900)

main()