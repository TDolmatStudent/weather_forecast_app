from datetime import date, timedelta
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time


SAVE_FILE_PATH = 'datasets/warsaw_weather_dataset.csv'
URL = 'https://www.pogodajutro.com/europe/poland/mazowieckie/warsaw?page=past-weather'


def main():
    driver = webdriver.Chrome()

    # Accepting cookies
    driver.get(f"{URL}#day=1&month=1")
    time.sleep(2)
    driver.find_element(By.XPATH, '/html/body/div[15]/div[2]/div[1]/div[2]/div[2]/button[1]').click()


    data = []

    start_date = date(2024, 1, 1) 
    day_count = 366 # including leap years (29 Feb)

    all_year_days = [start_date + timedelta(n) for n in range(day_count)]

    for date in all_year_days:
        day = date.day
        month = date.month

        driver.get(f"{URL}#day={day}&month={month}")

        measurements = driver.find_elements(By.XPATH, '//*[@id="years-table"]/div/table/tbody/tr')

        for measurement in measurements:
            year = measurement.find_element(By.XPATH, './td[1]/span[1]').text
            month_string = f'0{month}' if len(str(month)) == 1 else str(month)
            day_string = f'0{day}' if len(str(day)) == 1 else str(day)
            date = f'{year}-{month_string}-{day_string}'

            temp_max = measurement.find_element(By.XPATH, './td[2]/span[1]/span[1]').text
            temp_min = measurement.find_element(By.XPATH, './td[2]/span[2]/span[1]').text
            
            rain = measurement.find_element(By.XPATH, './td[3]/span[1]').text
            wind = measurement.find_element(By.XPATH, './td[6]/span[1]').text

            data.append([date, temp_max, temp_min, rain, wind])


    df = pd.DataFrame(data, columns=['date', 'temp_max', 'temp_min', 'rain', 'wind'])
    df.to_csv(SAVE_FILE_PATH, sep=';', index=False)

    driver.quit()


if __name__ == '__main__':
    main()
