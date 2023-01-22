################################################################################
# Scraping Single Wikipedia Page
################################################################################
# This page exctracts links from wikipedia page in a simplistic way:
from urllib import request
from bs4 import BeautifulSoup as BS
import random
import pandas as pd
import time
import csv

################################################################################
# This part prepares preliminary links - links for lists of links :)
################################################################################
url = 'https://www.lento.pl/motoryzacja/samochody.html' 
html = request.urlopen(url)
bs = BS(html.read(), 'html.parser')

no_pages = bs.find_all('span', {'class':'hash number alike button'})[0].text
int(no_pages)
links=[]

no_pages=274

for i in range(1,int(no_pages)):
    print(i)
    if i == 1:
        url = 'https://www.lento.pl/motoryzacja/samochody.html' 
        html = request.urlopen(url)
        bs = BS(html.read(), 'html.parser')
        tags = bs.find_all('a', {'class':'title-list-item'})
        links_page = [tag['href'] for tag in tags]
        #print(links_page)
    else:
        url = 'https://www.lento.pl/motoryzacja/samochody.html?page=' + str(i)
        html = request.urlopen(url)
        bs = BS(html.read(), 'html.parser')
        tags = bs.find_all('a', {'class':'title-list-item'})
        links_page = [tag['href'] for tag in tags]

    for x in links_page:
        links.append(x)

#print(len(links))
#print(links)
# ## "https://www.lento.pl/motoryzacja/samochody.html?page=2" -> takie format mają kolejne strony w kategorii samochodów 

# tags = bs.find_all('a', {'class':'title-list-item'}) ## Tutaj zgadniam HTML z każdej ze stron 1-n w kategorii samochodów

# links = [tag['href'] for tag in tags] ## Tutaj zgarniam linki do poszczególnych ofert -> stworzenie bazy linków 
# print(links)

data = pd.DataFrame({'offer_URL':[], 'promo_reg':[], 'promo_loc':[], 'location':[], 'date_added':[], 'seller_URL':[],
'car_category':[], 'car_type':[], 'car_gas_type':[], 'car_production_year':[], 'car_colour':[], 'car_technical_condition':[], 'car_gears_type':[], 
'car_country':[], 'car_mileage':[], 'car_eng_capacity':[], 'car_price':[], 'offer_header':[], 'offer_description':[], 'offer_vieved':[]})
bby=0
for link in links:
    rand_tim=random.random()
    time.sleep(rand_tim)
    bby=bby+1
    print(str(bby))
    print(link)
    try:
        html = request.urlopen(link, timeout = 10)
    except:
        print('**************************PROBLEM*********************************')
        continue
    bs_offer = BS(html.read(), 'html.parser')
    #print(bs_offer)

    tags_offer = bs_offer.find_all('div', {'class':'details text-15'})[0].find_all('span')
    tags_offer = bs_offer.find_all('span', {'class':'label'})
    tags_offer = bs_offer.find_all('span', {'class':'row-old'})
    #print(tags_offer)

    try:
        promo_reg = 0
        promo_loc = 0
        promo = bs_offer.find_all('div', {'class':'promo'})[0].text
        if promo ==' Promowane Regionalne' or promo == 'Promowane Regionalne' :
            promo_reg = 1
        elif promo ==' Promowane Lokalne' or promo =='Promowane Lokalne':
            promo_loc = 1
    except:
        promo = ''
        promo_reg = 0
        promo_loc = 0

    try:
       location = bs_offer.find_all('div', {'class':'userbox-location-city licon-pin-f'})[0].text
    except:
        location = ''

    try:
        date_added = bs_offer.find_all('div', {'class':'pull-left text-13'})[0].text
    except:
        date_added = ''

    try:
        table = bs_offer.find_all('div', {'class':'details text-15'})[0].find_all('li')
        car_country=''

        for i in range(0,len(table)):
            label = bs_offer.find_all('span', {'class':'label'})[i].text
            if label == 'Kategoria:':
                car_category = bs_offer.find('span',string = label).next_sibling.next_sibling.text

            if label == 'Typ nadwozia:':
                car_type = bs_offer.find('span',string = label).next_sibling.text

            if label == 'Paliwo:':
                car_gas_type = bs_offer.find('span',string = label).next_sibling.text

            if label == 'Rok produkcji:':
                car_production_year = bs_offer.find('span',string = label).next_sibling.text

            if label == 'Kolor:':
                car_colour = bs_offer.find('span',string = label).next_sibling.text

            if label == 'Stan techniczny:':
                car_technical_condition = bs_offer.find('span',string = label).next_sibling.text

            if label == 'Skrzynia biegów:':
                car_gears_type = bs_offer.find('span',string = label).next_sibling.text

            if label == 'Pochodzenie:':
                car_country = bs_offer.find('span',string = label).next_sibling.text

            if label == 'Przebieg:':
                car_mileage = bs_offer.find('span',string = label).next_sibling.text

            if label == 'Pojemność:':
                car_eng_capacity = bs_offer.find('span',string = label).next_sibling.text

            if label == 'Cena:':
                car_price = bs_offer.find('span',string = label).next_sibling.next_sibling.text
    except:
        table = ''

    try:
       offer_header = bs_offer.find('h2').text
    except:
        offer_header = ''

    try:
       offer_description = bs_offer.find_all('div', {'class':'desc text-15'})[0].text
    except:
        offer_description = ''

    try:
       offer_vieved = bs_offer.find_all('span', {'class':'text-b'})[0].text
    except:
        offer_vieved = ''

    offer = {'offer_URL':link, 'promo_reg':promo_reg, 'promo_loc':promo_loc, 'location':location, 'date_added':date_added, 'seller_URL':'x', 'car_category':car_category, 'car_type':car_type, 'car_gas_type':car_gas_type, 'car_production_year':car_production_year, 'car_colour':car_colour, 'car_technical_condition':car_technical_condition, 'car_gears_type':car_gears_type, 'car_country':car_country, 'car_mileage':car_mileage, 'car_eng_capacity':car_eng_capacity, 'car_price':car_price, 'offer_header':offer_header, 'offer_description':offer_description, 'offer_vieved':offer_vieved}

    data = data.append(offer, ignore_index = True)

data.to_csv('offersLento02012023.csv')
print(data)

c = pd.read_csv('C:/Users/justy/Desktop/Info/Inne/DSC/UW/Magisterka/offersLento02012023.csv')
data=data.drop_duplicates(subset=['offer_URL'],keep="first")
data=data.drop_duplicates(subset=['offer_header'],keep="first")
print(data)
## błąd w promo reg i loc !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Albo jednak jes ok