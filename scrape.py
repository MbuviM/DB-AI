import requests
from bs4 import BeautifulSoup
import json
import csv

# Define the URL of the website you want to scrape
url = "https://www.cdc.gov/diabetes/basics/diabetes.html"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content of the response
    soup = BeautifulSoup(response.content, "html.parser")
    # Find all the heading elements in the HTML
    headings = soup.find_all("h2")

    # Create a list to store the scraped data
    data = []

    # Extract the text from the heading elements and append to the data list
    for heading in headings:
        data.append(heading.text)

    # Define the path to the output CSV file
    output_file = "scraped.csv"

    # Write the data to the CSV file
    with open(output_file, "w", newline="") as file:
        csv.writer(file).writerow(data)

    # Print a success message
    print("Data scraped and stored in", output_file)

else:
    print("Failed to retrieve data. Status code:", response.status_code)
