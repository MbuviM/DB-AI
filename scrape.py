import requests
from bs4 import BeautifulSoup

def scrape_website():
    """
    Scrapes a website and saves the extracted text to a file.

    This function sends an HTTP request to a specified URL, retrieves the data, 
    and then uses BeautifulSoup to parse the HTML content. It extracts the text 
    from the website and saves it to a file named 'diabetes.txt'.
    """

    # Define the URL
    url = 'https://www.mayoclinic.org/diseases-conditions/diabetes/symptoms-causes/syc-20371444'

    # Define HTTP request
    data_retrieve = requests.get(url)

    # Check the status code of the request
    if data_retrieve.status_code == 200:
        print('Request successful')
    else:
        print('Request failed!')

    # Scrape the website
    soup = BeautifulSoup(data_retrieve.text, 'html.parser')

    # Extract the text from the website
    text = soup.select('p')

    # Print the text
    for word in text:
        print(word.get_text())

    # Save the text to a file
    with open('diabetes.txt', 'w') as file:
        for word in text:
            file.write(word.get_text())
            file.write('\n')

# Call the function to execute the code
scrape_website()
