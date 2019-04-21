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

	for movie_li in movie_list_soup.find_all('li'):
		detail = movie_li.find('div', attrs={'class': 'hd'})
		movie_name = detail.find('span', attrs={'class': 'title'}).getText()

		star = movie_li.find('div', attrs={'class': 'star'})
		movie_rate = star.find('span', attrs={'class': 'rating_num'}).getText()

		movie_name_list.append(movie_name)
		movie_rate_list.append(movie_rate)

	next_page = soup.find('span', attrs={'class': 'next'}).find('a')
	if next_page:
		return movie_name_list, movie_rate_list, DOWNLOAD_URL + next_page['href']
	return movie_name_list, movie_rate_list, None



url = DOWNLOAD_URL
df = pd.DataFrame(columns=['movies', 'rating'])
while url:
	html = download_page(url)
	movies, rates, url = parse_html(html)
	data = {}
	data['movies'] = movies
	data['rating'] = rates
	frame = pd.DataFrame(data)
	df = pd.concat([df, frame])

df.reset_index(drop=True, inplace=True)



'''
	with codecs.open('movie', 'wb', encoding='utf-8') as fp:
		while url:
						fp.write(u'{movies}\n'.format(movies='\n'.join(movies)))



def main():

	print(df)




if __name__ == '__main__':
	main()