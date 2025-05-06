'''get_turb_categories.py
This script iterates through the entire cleaned pireps database to grab all aircraft models.
It then pings an ITAO website to get it's turbulence category, first trying the complete aircraft
code and then trying the first thre letters of the aircraft code.
It then exports the results to a csv called plane_models_by_turb.csv.
'''
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time


#NOTE: We are only taking the first three values of the aircraft code

#grab unique aircraft codes from dictionary
plane_models = pd.read_csv('./plane_weight_dictionary.csv')['AIRCRAFT'].unique()
print(f'There are {len(plane_models)} unique plane models in this dataset.')


#Open chrome on webdriver (need to have selenium installed for this step)
browser = webdriver.Chrome()

# Open the ICAO website
browser.get("https://www.icao.int/publications/doc8643/pages/search.aspx")

# Wait for the page to load and locate the search bar
try:
    search_bar = WebDriverWait(browser, 10).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="atd-table"]/thead/tr/th[3]/input'))
    )
    print('Successfully found search bar.')
except Exception as e:
    print(f"Error locating search bar: {e}")
    browser.quit()
    exit()


#initialize empty dict to store plane models and turbulence categories
plane_models_by_turb = dict()

start_time = time.time()

first_time = True
for model in plane_models:
    try:
        #wait for website to start up
        if first_time:
            time.sleep(4)
            first_time = False

        # Clear the search bar and enter the plane model
        search_bar.clear()
        search_bar.send_keys(model)
        
        # Wait for the results to update
        time.sleep(2) 

        # Locate the first row of the results table
        first_row = WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="atd-table"]/tbody/tr[1]'))
        )
        
        # Access data from the result of text when querying website
        text = first_row.text

        # Find plane weight (Turbulence Category on ITAO website), and add it to dictionary
        plane_weight = text[-1]  # Assuming the last character indicates turbulence category

        '''since the plane weight variable stores the last category in the text, 
        if there is no plane found, it will store the last character in that resulting string, 
        which is 'no plane found'. Hence, it will assign a value 'd' to the plane weight variable.
        We cast this to 'U' to better represent an unknown plane weight.'''

        # Retry if the plane weight category is 'd'
        if plane_weight == 'd':
            print(f"Retrying with the first three letters for {model}.")
            search_bar.clear()
            search_bar.send_keys(model[:3])
            
            # Wait for the results to update
            time.sleep(2)

            # Locate the first row again
            first_row = WebDriverWait(browser, 10).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="atd-table"]/tbody/tr[1]'))
            )
            text = first_row.text
            plane_weight = text[-1]  # Update turbulence category

        # Add to dictionary
        plane_models_by_turb[model] = plane_weight

    except Exception as e:
        # Default turbulence category to "U" if no results or an error occurs
        print(f"Error fetching results for {model}: {e}")
        plane_models_by_turb[model] = "U"

browser.quit()

end_time = time.time()
print(plane_models_by_turb)

# Convert dictionary to DataFrame
plane_models_df = pd.DataFrame(list(plane_models_by_turb.items()), columns=['Aircraft', 'Turbulence_Category'])

# Save the DataFrame to a CSV file
plane_models_df['Turbulence_Category'] = plane_models_df['Turbulence_Category'].replace('d', 'U')

plane_models_df.to_csv('plane_weight_dictionary.csv', index=False)

print("Dictionary successfully saved to 'plane_models_by_turb.csv'.")

print('total time is', end_time - start_time)
