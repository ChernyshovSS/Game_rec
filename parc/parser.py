import requests
from bs4 import BeautifulSoup
import pandas as pd


url = "https://iwant.games/games/"

response = requests.get(url)
data = BeautifulSoup(response.text, "lxml")


game_info = []

for page in range(1, 67):
    page_url = url + f"page/{page}/"
    print(f"Парсинг страницы: {page_url}")

    response = requests.get(page_url)
    data = BeautifulSoup(response.text, "lxml")


    for game_data in data.find_all('article',class_='game'):
        game_title = game_data.find('h2').text
        image_path = game_data.find('a')
        game_img = game_data.find('img', class_="attachment-game size-game wp-post-image")['src']
        game_url = image_path['href']
        
            # Заходим на страницу фильма и извлекаем описание
        game_description = requests.get(game_url)
        game_page = BeautifulSoup(game_description.text, "lxml")
        
        try:
            description = (
                game_page.find("div", class_="single__main").find("p").text
            )
        except AttributeError:
            description = "Описание не найдено"

        game_info.append(
            {
                "title": game_title,
                "image_url": game_img,
                "page_url": game_url,
                "description": description,
            }
        )
        print(game_title, '\n', game_img, '\n', game_url, '\n',description,'\n')
        
df = pd.DataFrame(game_info)
df.to_csv('game_data.csv', index=False)