#Install libraries
from bs4 import BeautifulSoup
import requests as re

# Request to get the data
url = input("Enter url: ")
def get_request():
    try:
        access_data = re.get(url=url)
        if access_data.status_code == 200:
            return access_data.content
        else:
            print(f"Error found! Received status code {access_data.status_code}")
            return None
    except Exception as e:
        print(f"An error occured: {e}")
    
def scrape_data(content):
    # Parse the content
    soup = BeautifulSoup(content, 'html.parser')
    #print(soup.prettify())

    # Scrape the data and store it
    with open("data-1.txt", "a", encoding="utf-8") as file:
        for text in soup.find_all(class_="content"):
            file.write(text.get_text() + "\n")


if __name__ == "__main__":

    content = get_request()
    if content is None:
        print("No data was scraped")
    else:
        scrape_data(content)
        print("Process was successful.")
