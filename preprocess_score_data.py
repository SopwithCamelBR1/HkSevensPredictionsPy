import csv
from pyquery import PyQuery as pq
import pandas as pd

rawdata_filename = "Data/world_rugby_sevens_series.csv"
data_filename = "Data/world_rugby_sevens_score_data.csv"
clean_data_filename = "Data/world_rugby_sevens_score_clean_data.csv"

html_array = []
data_array = []

#get html data from raw file
with open(rawdata_filename, newline='') as csvfile:
  csvreader = csv.reader(csvfile)
  for row in csvreader:
    #print(row[6])
    html_array.append(row[6])

#provess data    
for item in html_array:
  html = item
  d = pq(html)
  info = d('div.info').text().replace(',', ' -')
  teams = d('div.teamName.left').text().replace('7s ', ' 7s, ')
  #teamB = d('div.teamName.left').text()
  scoreA, sep, scoreB = d('div.result.left').text().partition(' - ')
  venue = d('div.info.info--venue').text().replace(',', ' -')
  data = teams + ', ' + scoreA + ', ' + scoreB + ', ' # + info + ', ' + venue  + ', '
  #print(data)
  data_array.append(data)
  
#print(data_array)

#write processed data to csv
with open(data_filename, 'w', newline='') as csvfile:
  csvwriter = csv.writer(csvfile)
  header = "teamA, teamB, scoreA, scoreB," #Event, Stage, Match, venue, 
  print(header)
  csvwriter.writerow([header])
  for row in data_array:
    print(row)
    csvwriter.writerow([row])
  print(header)

#remove empty lines
data = pd.read_csv(data_filename)
print(data)
data.dropna().to_csv(clean_data_filename)