import csv
import pdb
import requests
from bs4 import BeautifulSoup

outputfile = open("./india_holidays.csv", "a")
writer = csv.writer(outputfile)
country = ["india"]
for cont in country:
    for year in range(2013, 2018):
        # url = "https://www.timeanddate.com/holidays/" + cont + "/" + str(year)
        url = "http://www.officeholidays.com/countries/" + cont + "/" + str(year) + ".php"
        response = requests.get(url)
        html = response.content
        soup = BeautifulSoup(html)
        tables = soup.findAll('table', attrs={'class', 'list-table'})
        # pdb.set_trace()
        for table in tables:
            for row in  table.findAll('tr'):
                list_of_cells = []
                list_of_cells.append(year)
                list_of_cells.append(cont)
                # for cell in row.findAll('th'):
                #     text = cell.text.replace('&nbsp', '')
                #     list_of_cells.append(text)
                for cell in row.findAll('td'):
                    text = cell.text.replace('&nbsp', '').replace('\n', '')
                    list_of_cells.append(text)
                writer.writerow(list_of_cells)
