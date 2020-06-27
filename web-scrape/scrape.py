import csv
import pdb
import requests
from bs4 import BeautifulSoup

outputfile = open("./holidays.csv", "a")
writer = csv.writer(outputfile)
country = ["afghanistan","albania","algeria","american-samoa","andorra","angola","anguilla","antigua-and-barbuda","argentina","armenia","aruba","australia","austria","azerbaijan","bahrain","bangladesh","barbados","belarus","belgium","belize","benin","bermuda","bhutan","bolivia","bosnia","botswana","brazil","british-virgin-islands","brunei","bulgaria","burkina-faso","burundi","cape-verde","cambodia","cameroon","canada","cayman-islands","central-african-republic","chad","chile","china","colombia","comores","republic-of-the-congo","dr-congo","cook-islands","costa-rica","ivory-coast","croatia","cuba","curacao","cyprus","czech","denmark","djibouti","dominica","dominican-republic","timor-leste","ecuador","egypt","el-salvador","guineaecuatorial","eritrea","estonia","ethiopia","falkland-islands","faroe-islands","fiji","finland","france","french-polynesia","gabon","gambia","georgia","germany","ghana","gibraltar","greece","greenland","grenada","guam","guatemala","guernsey","guinea","guinea-bissau","guyana","haiti","vatican-city-state","honduras","hong-kong","hungary","iceland","india","indonesia","iran","iraq","ireland","israel","italy","jamaica","japan","jersey","jordan","kazakhstan","kenya","kiribati","kosovo","kuwait","kyrgyzstan","laos","latvia","lebanon","lesotho","liberia","libya","liechtenstein","lithuania","luxembourg","macau","macedonia","madagascar","malawi","malaysia","maldives","mali","malta","marshall-islands","martinique","mauritania","mauritius","mayotte","mexico","micronesia","moldova","monaco","mongolia","montenegro","montserrat","morocco","mozambique","myanmar","namibia","nauru","nepal","netherlands","new-caledonia","new-zealand","nicaragua","niger","nigeria","north-korea","northern-mariana-islands","norway","oman","pakistan","palau","panama","papua-new-guinea","paraguay","peru","philippines","poland","portugal","puerto-rico","qatar","reunion","romania","russia","rwanda","saint-helena","saint-kitts-and-nevis","saint-lucia","saint-martin","saint-pierre-and-miquelon","saint-vincent-and-the-grenadines","samoa","san-marino","sao-tome-and-principe","saudi-arabia","senegal","serbia","seychelles","sierra-leone","singapore","sint-maarten","slovakia","slovenia","solomon-islands","somalia","south-africa","south-korea","south-sudan","spain","sri-lanka","saint-barthelemy","sudan","suriname","swaziland","sweden","switzerland","syria","taiwan","tajikistan","tanzania","thailand","bahamas","togo","tonga","trinidad","tunisia","turkey","turkmenistan","turks-and-caicos-islands","tuvalu","united-states-virgin-islands","uganda","ukraine","united-arab-emirates","uk","us","uruguay","uzbekistan","vanuatu","venezuela","vietnam","wallis-and-futuna","yemen","zambia","zimbabwe","un","world"]
for cont in country:
    for year in range(2046, 2105):
        url = "https://www.timeanddate.com/holidays/" + cont + "/" + str(year)
        response = requests.get(url)
        html = response.content
        soup = BeautifulSoup(html)
        table = soup.find('table', attrs={'class', 'zebra'})
        for row in  table.findAll('tr'):
            list_of_cells = []
            list_of_cells.append(year)
            list_of_cells.append(cont)
            for cell in row.findAll('th'):
                text = cell.text.replace('&nbsp', '')
                list_of_cells.append(text)
            for cell in row.findAll('td'):
                text = cell.text.replace('&nbsp', '')
                list_of_cells.append(text)
            writer.writerow(list_of_cells)
