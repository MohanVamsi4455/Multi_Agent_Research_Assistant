from src.tools.tools import web_search ,scrape_url

# output=web_search.invoke("News on AI Research")

# print(output)


scrape_result=scrape_url.invoke("https://www.artificialintelligence-news.com/")
print(scrape_result)




