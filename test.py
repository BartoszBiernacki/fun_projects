import abc
import pathlib
import pytube
import os
import re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import compress
import time
import yaml
import glob
from bs4 import BeautifulSoup
import requests
import time
import threading

import moviepy.editor as editor


def get_clip_urls_from_playlist_url(playlist_url: str) -> list[str]:
    return list(pytube.Playlist(playlist_url).video_urls)


def get_video_titles_from_playlist(playlist_url: str) -> list[str]:
    return [video.title for video in pytube.Playlist(playlist_url).videos]


def get_names_of_already_download_clips(
        path: str, prefix_len: int) -> list[str]:
    if os.path.exists(path):
        return [os.path.splitext(fname[prefix_len:])[0]
                for fname in os.listdir(path)]


def get_missing_uls(path: str, prefix_len: int) -> list[str]:
    online_names = get_video_titles_from_playlist()
    downloaded_names = get_names_of_already_download_clips(output_path,
                                                           current_prefix_len)


def get_video_title(url):
    # Make a GET request to the video's URL
    response = requests.get(url)

    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the title element in the HTML
    title_element = soup.find('title')

    # Extract the text from the title element
    title = title_element.text

    # Return the title
    return title


if __name__ == '__main__':
    current_prefix_len = 4
    output_path = r'C:\Users\HAL\Downloads\YouTube2\audio\The best of Szkolna17'
    url_US17 = 'https://www.youtube.com/playlist?list=PLda6VETsZ3sALHJVBmefYVvZXR9Gp-lqu'
    url30 = 'https://www.youtube.com/watch?v=Ltet3RR8XC0&list=PLda6VETsZ3sALHJVBmefYVvZXR9Gp-lqu&index=30'
    url31 = 'https://www.youtube.com/watch?v=4Hc9D89poyw&list=PLda6VETsZ3sALHJVBmefYVvZXR9Gp-lqu&index=31'


# 18s
for video in pytube.Playlist(url_US17).videos:
    title = video.title


start_time = time.perf_counter()
threads = []
for video in pytube.Playlist(url_US17).videos:
    thread = threading.Thread(target=video.title)
    threads.append(thread)

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()
elapsed_time = time.perf_counter() - start_time
print(f"Execution time: {elapsed_time:.6f} seconds")

for num_workers in [16, 32]:
    start_time = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=num_workers)
    urls = pytube.Playlist(url_US17).video_urls
    titles = list(executor.map(get_video_title, urls))
    elapsed_time = time.perf_counter() - start_time
    print(f"[{num_workers}] Execution time: {elapsed_time:.6f} seconds")


# def get_default_filenames_from_playlist(playlist_url: str) -> list[str]:
#     return [pytube.YouTube(url, use_oauth=True, allow_oauth_cache=True)
#             .streams[0].default_filename
#             for url in pytube.Playlist(playlist_url).video_urls]

# name = audio_stream.download(output_path, prefix)

# get_audio_stream(url30)
# get_audio_stream(url31)
