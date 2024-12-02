import time
import json
import requests
import tkinter as tk
from bs4 import BeautifulSoup
# 访问头设置
headers={'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
             'Accept-Encoding': 'gzip, deflate',
             'Accept-Language': 'zh-CN,zh;q=0.9',
             'Connection': 'keep-alive',
             'Cookie': '_iuqxldmzr_=32; _ntes_nnid=0e6e1606eb78758c48c3fc823c6c57dd,1527314455632; '
                       '_ntes_nuid=0e6e1606eb78758c48c3fc823c6c57dd; __utmc=94650624; __utmz=94650624.1527314456.1.1.'
                       'utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); WM_TID=blBrSVohtue8%2B6VgDkxOkJ2G0VyAgyOY;'
                       ' JSESSIONID-WYYY=Du06y%5Csx0ddxxx8n6G6Dwk97Dhy2vuMzYDhQY8D%2BmW3vlbshKsMRxS%2BJYEnvCCh%5CKY'
                       'x2hJ5xhmAy8W%5CT%2BKqwjWnTDaOzhlQj19AuJwMttOIh5T%5C05uByqO%2FWM%2F1ZS9sqjslE2AC8YD7h7Tt0Shufi'
                       '2d077U9tlBepCx048eEImRkXDkr%3A1527321477141; __utma=94650624.1687343966.1527314456.1527314456'
                       '.1527319890.2; __utmb=94650624.3.10.1527319890',
             'Host': 'music.163.com',
             'Referer': 'http://music.163.com/',
             'Upgrade-Insecure-Requests': '1',
             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                           'Chrome/66.0.3359.181 Safari/537.36'}
# 歌手页码id和偏移量
ids = [1001, 1002, 1003, 2001, 2002, 2003, 6001, 6002, 6003, 7001, 7002, 7003, 4001, 4002, 4003]
inistals = [65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
       79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 0]
# 在网易云官网查找歌手名称和id
def get_artistid(str_find:str) -> int:
    artist_id = -1
    try:
        for id in ids:
            for inistal in inistals:
                url_artist = 'http://music.163.com/discover/artist/cat?id={}&initial={}'.format(str(id), str(inistal))
                response_artist = requests.get(url_artist, headers=headers)
                time.sleep(0.02)
                soup_artist = BeautifulSoup(response_artist.text, 'html.parser')
                for artist in soup_artist.find_all('a', attrs={'class':'nm nm-icn f-thide s-fc0'}):
                    artist_name = artist.string
                    artist_id = artist['href'].replace('/artist?id=', '').strip()
                    if str_find in artist_name:
                        raise StopIteration
    except StopIteration:
        pass
    finally:
        return artist_id
# 通过歌手id查找歌手的歌曲列表
def get_songid(str_find:str, artist_id:int) -> tuple:
    song_id = -1
    song_name = ''
    url_songs = 'https://music.163.com/api/v1/artist/songs'
    params_songs = {'id': artist_id,
              'offset': 0,
              'total': True,
              'limit': 1000}
    response_songs = requests.get(url_songs, headers=headers, params=params_songs)
    time.sleep(0.02)
    result_songs = json.loads(response_songs.text)
    songs = result_songs['songs']
    try:
        for _,song in enumerate(songs):
            if str_find in song['name']:
                song_name = song['name']
                song_id = song['id']
                raise StopIteration
    except StopIteration:
        pass
    finally:
        return song_id, song_name
# 通过歌曲id下载歌曲
def download(song_id:int, song_name:str) -> None:
    url_down = 'https://music.163.com/song/media/outer/url?id={}'.format(song_id)
    response_down = requests.get(url_down)
    with open(song_name+'.mp3', 'wb') as f:
            f.write(response_down.content)
# 歌手名称确认按键回调函数
def main():
    artist_id = get_artistid(ent_artist_name.get())
    if artist_id == -1:
        tk.Label(win, text='未查询到该歌手').grid(row=4, column=0)
    else:
        song_id, song_name = get_songid(ent_song_name.get(), artist_id)
        if song_id == -1:
            tk.Label(win, text='未查询到首歌  ').grid(row=4, column=0)
        else:
            download(song_id, song_name)
            tk.Label(win, text='下载成功     ').grid(row=4, column=0)
# 窗口模块初始化
win = tk.Tk()
win.title('网易云音乐下载器')
win.geometry('300x100')
win.resizable(False, False)
# 歌手名称输入设置
tk.Label(win, text='请输入歌手名称:', font=('宋体', 12)).grid(row=0, column=0)
tk.Label(win, text='请输入想下的歌:', font=('宋体', 12)).grid(row=1, column=0)
ent_artist_name = tk.Entry(win, font=('宋体', 10))
ent_artist_name.grid(row=0, column=1)
ent_song_name = tk.Entry(win, font=('宋体', 10))
ent_song_name.grid(row=1, column=1)
# 确认按键设置
tk.Button(win, text='确认', font=('宋体', 12), command=main).grid(row=3, column=1)
# 设置显示列表框
# songs_list = tk.Listbox(win, width=71).place(x=0, y=40)

tk.mainloop()