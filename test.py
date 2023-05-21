import abc
import pathlib
import random

import pytube
import os
import re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Process, Queue, current_process
from itertools import compress
import time
import glob
from bs4 import BeautifulSoup
import requests
import time
import threading
from dataclasses import dataclass
import moviepy.editor as editor
from threading import Thread, current_thread


def get_clip_urls_from_playlist_url(playlist_url: str) -> list[str]:
    return list(pytube.Playlist(playlist_url).video_urls)


def get_names_of_already_download_clips(
        path: str, prefix_len: int) -> list[str]:
    if os.path.exists(path):
        return [os.path.splitext(fname[prefix_len:])[0]
                for fname in os.listdir(path)]


def get_missing_uls(path: str, prefix_len: int) -> list[str]:
    online_names = get_video_titles_from_playlist()
    downloaded_names = get_names_of_already_download_clips(output_path,
                                                           current_prefix_len)


def get_video_title(url: str) -> str:
    # Make a GET request to the video's URL
    response = requests.get(url)

    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the title element in the HTML
    title_element = soup.find('title')

    # Extract the text from the title element
    title = title_element.text

    # Return the title without at the end ' - youtube'
    return title[:-10]


def get_video_titles_from_playlist(url: str, num_workers=32) -> list[str]:
    executor = ThreadPoolExecutor(max_workers=num_workers)
    urls = pytube.Playlist(url).video_urls
    titles = list(executor.map(get_video_title, urls))
    return titles


def remove_illegal_characters(filename: str) -> str:
    illegal_characters = r'[\\/:*?"<>|]'
    return re.sub(illegal_characters, '', filename)


if __name__ == '__main__':
    current_prefix_len = 4
    output_path = r'C:\Users\HAL\Downloads\YouTube2\audio\The best of Szkolna17'
    url_US17 = 'https://www.youtube.com/playlist?list=PLda6VETsZ3sALHJVBmefYVvZXR9Gp-lqu'
    url30 = 'https://www.youtube.com/watch?v=Ltet3RR8XC0&list=PLda6VETsZ3sALHJVBmefYVvZXR9Gp-lqu&index=30'
    url31 = 'https://www.youtube.com/watch?v=4Hc9D89poyw&list=PLda6VETsZ3sALHJVBmefYVvZXR9Gp-lqu&index=31'



start_time = time.perf_counter()
l2 = get_video_titles_from_playlist(url=url_US17, num_workers=12)
end_time = time.perf_counter()
print(f'[smart] Executed in {end_time - start_time:0.6f} seconds')


# def get_default_filenames_from_playlist(playlist_url: str) -> list[str]:
#     return [pytube.YouTube(url, use_oauth=True, allow_oauth_cache=True)
#             .streams[0].default_filename
#             for url in pytube.Playlist(playlist_url).video_urls]

# name = audio_stream.download(output_path, prefix)

# get_audio_stream(url30)
# get_audio_stream(url31)
