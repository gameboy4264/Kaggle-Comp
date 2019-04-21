#!/user/bin/env python
# encoding=utf-8

import requests
from bs4 import BeautifulSoup
import codecs
import pandas as pd

DOWNLOAD_URL = 'http://movie.douban.com/top250'

def download_page(url):
	data = requests.get(url).content
	return data



def parse_html(html):

	soup = BeautifulSoup(html)
	movie_list_soup = soup.find('ol', attrs={'class': 'grid_view'})

	movie_name_list = []
	movie_rate_list = []
	movie_info_list = []

	for movie_li in movie_list_soup.find_all('li'):
		detail = movie_li.find('div', attrs={'class': 'hd'})
		movie_name = detail.find('span', attrs={'class': 'title'}).getText()

		star = movie_li.find('div', attrs={'class': 'star'})
		movie_rate = star.find('span', attrs={'class': 'rating_num'}).getText()

		info = movie_li.find('div', attrs={'class': 'bd'})
		movie_info = info.find('p').getText()

		movie_name_list.append(movie_name)
		movie_rate_list.append(movie_rate)
		movie_info_list.append(movie_info)

	next_page = soup.find('span', attrs={'class': 'next'}).find('a')
	if next_page:
		return movie_name_list, movie_rate_list, movie_info_list, DOWNLOAD_URL + next_page['href']
	return movie_name_list, movie_rate_list, movie_info_list, None



url = DOWNLOAD_URL
df = pd.DataFrame(columns=['movies', 'rating', 'infomation'])
while url:
	html = download_page(url)
	movies, rates, info, url = parse_html(html)
	data = {}
	data['movies'] = movies
	data['rating'] = rates
	data['infomation'] = info
	frame = pd.DataFrame(data)
	df = pd.concat([df, frame])

df.reset_index(drop=True, inplace=True)


df['year'] = df.infomation.apply(lambda x: x.split('\n')[2].strip().split('\xa0')[0])
df['country'] = df.infomation.apply(lambda x: x.split('\n')[2].strip().split('\xa0')[2])
df['genr'] = df.infomation.apply(lambda x: x.split('\n')[2].strip().split('\xa0')[4])
df['director'] = df.infomation.apply(lambda x: x.split('\n')[1].strip().split('\xa0')[0].split(':')[1])
df['actor'] = ""
for a in range(250):
    try:
        df['actor'][a] = df['infomation'][a].split('\n')[1].strip().split('\xa0')[3].split(':')[1]
    except:
        df['actor'][a] = 'None'
df.drop('infomation', axis=1, inplace=True)

df.to_csv('movie.csv')
