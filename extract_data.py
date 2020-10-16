import bs4 as bs
import urllib.request

def Extract_Data(url):
	print('Extracting...')
	article = urllib.request.urlopen(url)
	parsed_article = bs.BeautifulSoup(article, 'lxml')
	paragraphs = parsed_article.find_all('p')
	raw_text = u""
	for p in paragraphs:
		raw_text += p.text

	print('Extraction Complete')
	return raw_text

