from __future__ import annotations

from dataclasses import dataclass
import os
import re
import pytube
from multiprocessing import Process, Value
from multiprocessing import Queue as Queue_mp  # Queue for multiprocessing
from queue import Queue as Queue_mth  # Queue for multithreading
from threading import Thread
import moviepy.editor as editor
import glob
from bs4 import BeautifulSoup
import requests
from concurrent.futures import ThreadPoolExecutor
import http.client


@dataclass
class UrlAndIdx:
    url: str
    idx: int


@dataclass
class OnlineClipInfo:
    url: str
    prefix: str = None
    title: str = None
    default_filename: str = None
    is_downloaded: bool = None


@dataclass
class DownloadedClipInfo:
    file_directory: str = None
    filename: str = None
    is_converted: bool = None

    def __post_init__(self):
        if self.file_directory is not None:
            if not self.filename:
                self.filename = os.path.split(self.file_directory)[1]
            if self.is_converted is None:
                self.is_converted = self.file_directory.endswith('.mp3')


class YtUrlTypeRecognizer:

    @staticmethod
    def is_video(url: str) -> bool:
        video_pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
        regex = re.compile(video_pattern)
        function_match = regex.search(url)
        return bool(function_match)

    @staticmethod
    def is_channel(url: str) -> bool:
        channel_patterns = [
            r"(?:\/(c)\/([%\d\w_\-]+)(\/.*)?)",
            r"(?:\/(channel)\/([%\w\d_\-]+)(\/.*)?)",
            r"(?:\/(u)\/([%\d\w_\-]+)(\/.*)?)",
            r"(?:\/(user)\/([%\w\d_\-]+)(\/.*)?)"
        ]
        for pattern in channel_patterns:
            regex = re.compile(pattern)
            function_match = regex.search(url)
            if function_match:
                return True
        return False

    @staticmethod
    def is_playlist(url: str) -> bool:
        playlist_pattern = r'www\.youtube\.com/playlist\?list='
        regex = re.compile(playlist_pattern)
        function_match = regex.search(url)
        return bool(function_match)

    @classmethod
    def get_clip_urls(cls, urls: list[str]) -> list[str]:
        return [url for url in urls if cls.is_video(url)]

    @classmethod
    def get_channel_urls(cls, urls: list[str]) -> list[str]:
        return [url for url in urls if cls.is_channel(url)]

    @classmethod
    def get_playlist_urls(cls, urls: list[str]) -> list[str]:
        return [url for url in urls if cls.is_playlist(url)]


class YtUrlGetter:

    @staticmethod
    def get_prefix(url_idx: int, num_of_all_urls: int) -> str:
        """ Returns prefix (str) like `[001]` `[002]` and so on.

        Useful for sorting downloaded files by order of their appearance on
        YouTube.
        """
        num_of_digits = len(str(num_of_all_urls)) + 1
        return f"[{str(url_idx + 1).zfill(num_of_digits)}]"

    @classmethod
    def put_online_clip_info_to_queue(
            cls,
            url_and_idx_queue: Queue_mth[UrlAndIdx],
            num_of_all_urls,
            downloaded_fnames: list[str],
            queue: Queue_mth[OnlineClipInfo],
            num_of_active_scrapers: Value('i'),
    ) -> None:

        while not url_and_idx_queue.empty():
            # print(f"[{num_of_active_scrapers.value}] active scrapers")
            url_and_idx = url_and_idx_queue.get()
            url = url_and_idx.url
            url_idx = url_and_idx.idx

            title = cls.get_video_title(url=url)
            online_name = cls.get_default_filename(title)
            queue.put(
                OnlineClipInfo(
                    url=url,
                    prefix=cls.get_prefix(
                        url_idx=url_idx,
                        num_of_all_urls=num_of_all_urls,
                    ),
                    title=title,
                    default_filename=cls.get_default_filename(title),
                    is_downloaded=online_name in downloaded_fnames,
                )
            )

        with num_of_active_scrapers.get_lock():
            if num_of_active_scrapers.value == 1:
                queue.put(YtDownloader.ONLINE_SENTINEL)
            num_of_active_scrapers.value -= 1

    @classmethod
    def put_playlist_clips_info_to_queue(
            cls,
            num_of_scraper_workers: int,
            playlist_url: str,
            downloaded_fnames: list[str],
            queue: Queue_mth[OnlineClipInfo],
    ) -> None:

        print('Putting OnlineClipInfo info to queue...')
        urls = cls.clip_urls_from_playlist(playlist_url)

        # prepare queue for scraper workers
        url_idx_queue = Queue_mth()
        for idx, url in enumerate(urls):
            url_idx_queue.put(UrlAndIdx(url=url, idx=idx))

        # keep num of all working scrapers to put only one sentinel at the end
        num_of_active_scrapers = Value('i', 0)

        # create scraper workers
        for _ in range(num_of_scraper_workers):
            with num_of_active_scrapers.get_lock():
                num_of_active_scrapers.value += 1

            scraper = Thread(
                target=cls.put_online_clip_info_to_queue,
                kwargs={
                    'url_and_idx_queue': url_idx_queue,
                    'num_of_all_urls': len(urls),
                    'downloaded_fnames': downloaded_fnames,
                    'queue': queue,
                    'num_of_active_scrapers': num_of_active_scrapers,
                }
            )
            scraper.start()

    @staticmethod
    def put_downloaded_clip_info_to_queue(
            path: str,
            queue: Queue_mp[DownloadedClipInfo],
    ) -> None:
        print('Putting DownloadedClipInfo info to queue...')

        if os.path.exists(path):
            for fdir in glob.glob(os.path.join(path, r'*.*')):
                queue.put(DownloadedClipInfo(file_directory=fdir))

    @staticmethod
    def clip_urls_from_playlist(playlist_url: str) -> list[str]:
        yt_playlist = pytube.Playlist(playlist_url)
        return list(yt_playlist.video_urls)

    @staticmethod
    def get_clip_urls_from_channel_url(channel_url: str) -> list[str]:
        yt_channel = pytube.Channel(channel_url)
        return list(reversed(yt_channel.video_urls))

    @staticmethod
    def get_audio_stream(clip_url: str) -> pytube.streams.Stream:
        audio_streams = pytube.YouTube(
            clip_url,
            use_oauth=True,
            allow_oauth_cache=True).streams.filter(only_audio=True)
        return audio_streams.order_by('abr')[-1]

    @staticmethod
    def get_playlist_name(playlist_url: str) -> str:
        return pytube.Playlist(playlist_url).title

    @staticmethod
    def get_channel_name(channel_url: str) -> str:
        return pytube.Channel(channel_url).title

    @staticmethod
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

    @classmethod
    def get_video_titles_from_playlist(
            cls, url: str, num_workers=32) -> list[str]:

        print('Getting video titles from playlist...')
        executor = ThreadPoolExecutor(max_workers=num_workers)
        urls = pytube.Playlist(url).video_urls
        titles = list(executor.map(cls.get_video_title, urls))
        return titles

    @staticmethod
    def get_default_filename(title: str) -> str:
        illegal_characters = r'[\\/:*?"<>|]'
        return re.sub(illegal_characters, '', title)

    @classmethod
    def get_default_filenames(cls, titles: list[str]) -> list[str]:
        print('Getting default filenames')
        return [cls.get_default_filename(title) for title in titles]

    @staticmethod
    def get_prefix_len(url: str) -> int:
        if YtUrlTypeRecognizer.get_playlist_urls([url]):
            return len(str(pytube.Playlist(url).length)) + 3
        elif YtUrlTypeRecognizer.get_channel_urls([url]):
            return len(str(pytube.Channel(url).length)) + 3
        else:
            return 0


class YtDownloader:
    ONLINE_SENTINEL = OnlineClipInfo(url='')
    DOWNLOADED_SENTINEL = DownloadedClipInfo()

    def __init__(self, urls: list[str]):

        self.playlist_urls = YtUrlTypeRecognizer.get_playlist_urls(urls)
        self.channel_urls = YtUrlTypeRecognizer.get_channel_urls(urls)
        self.clip_urls = YtUrlTypeRecognizer.get_clip_urls(urls)

        self.output_base_path = os.path.join(self._get_output_base_path())
        self.num_of_downloader_workers = 20
        self.num_of_scraper_workers = 30

    @staticmethod
    def _get_output_base_path() -> str:
        # Expand the '~' symbol to the path of the user's home directory
        home_dir = os.path.expanduser('~')
        # Return the default download folder path
        return os.path.join(home_dir, 'Downloads', 'YouTube2', 'audio')

    @staticmethod
    def remove_corrupted_files(path: str):
        if os.path.exists(path):
            for fdir in glob.glob(os.path.join(path, r'*.*')):
                if os.path.getsize(fdir) == 0:
                    os.remove(path=fdir)

    @staticmethod
    def create_prefix(url_idx: int, num_of_all_urls: int) -> str:
        """ Returns prefix (str) like `[001]` `[002]` and so on.

        Useful for sorting downloaded files by order of their appearance on
        YouTube.
        """
        num_of_digits = len(str(num_of_all_urls)) + 1
        return f"[{str(url_idx + 1).zfill(num_of_digits)}]"

    def get_output_path(self, url: str) -> str:
        if YtUrlTypeRecognizer.get_playlist_urls([url]):
            return os.path.join(
                self.output_base_path,
                YtUrlGetter.get_playlist_name(url))
        elif YtUrlTypeRecognizer.get_channel_urls([url]):
            return os.path.join(
                self.output_base_path,
                YtUrlGetter.get_channel_name(url))
        else:
            return os.path.join(
                self.output_base_path,
                'random clips')

    @staticmethod
    def get_names_of_already_download_clips(
            path: str, prefix_len: int) -> list[str]:
        print('Getting names of already downloaded clips ...')

        if os.path.exists(path):
            return [os.path.splitext(fname[prefix_len:])[0]
                    for fname in os.listdir(path)]
        return []

    @classmethod
    def download_audio(
            cls,
            online_clips_info: Queue_mth[OnlineClipInfo],
            downloaded_clips_info: Queue_mp[DownloadedClipInfo],
            num_of_active_downloaders: Value('i'),
            output_path: str,
    ) -> None:

        while True:
            online_clip_info = online_clips_info.get()

            if online_clip_info == cls.ONLINE_SENTINEL:
                online_clips_info.put(cls.ONLINE_SENTINEL)
                break

            if not online_clip_info.is_downloaded:
                audio_stream = YtUrlGetter.get_audio_stream(
                    online_clip_info.url
                )
                ext = os.path.splitext(audio_stream.default_filename)[1]
                print(f"[downloading] {online_clip_info.prefix}"
                      f" {online_clip_info.default_filename + ext}")

                try:
                    fdir = audio_stream.download(
                        output_path=output_path,
                        filename=online_clip_info.default_filename + ext,
                        filename_prefix=online_clip_info.prefix
                    )
                    downloaded_clips_info.put(
                        DownloadedClipInfo(
                            file_directory=fdir,
                            filename=os.path.split(fdir)[1],
                            is_converted=os.path.splitext(fdir)[1] == '.mp3'
                        )
                    )
                except http.client.IncompleteRead:
                    # Try again later
                    online_clips_info.put(online_clip_info)

        with num_of_active_downloaders.get_lock():
            if num_of_active_downloaders.value == 1:
                downloaded_clips_info.put(cls.DOWNLOADED_SENTINEL)
            num_of_active_downloaders.value -= 1

    @classmethod
    def convert_to_mp3_and_remove(
            cls,
            downloaded_clips_info: Queue_mp[DownloadedClipInfo]
    ) -> None:

        while True:
            clip_info = downloaded_clips_info.get()
            if clip_info == cls.DOWNLOADED_SENTINEL:
                downloaded_clips_info.put(cls.DOWNLOADED_SENTINEL)
                break

            if not clip_info.is_converted:
                print(f"[mp3 converting] {clip_info.filename}", flush=True)
                audio_clip = editor.AudioFileClip(clip_info.file_directory)

                audio_clip.write_audiofile(
                    os.path.splitext(clip_info.file_directory)[0] + '.mp3',
                    verbose=False,
                    logger=None,
                )
                audio_clip.close()
                os.remove(clip_info.file_directory)
            else:
                print(f"[mp3 exist] {clip_info.filename}", flush=True)

    def download_playlist(self, playlist_url: str) -> None:
        playlist_name = YtUrlGetter.get_playlist_name(playlist_url)
        print(f'Downloading playlist: {playlist_name}')

        output_path = self.get_output_path(playlist_url)
        self.remove_corrupted_files(path=output_path)

        # Create `online_clips_info` queue
        online_clips_info = Queue_mth()
        YtUrlGetter.put_playlist_clips_info_to_queue(
            num_of_scraper_workers=self.num_of_scraper_workers,
            playlist_url=playlist_url,
            downloaded_fnames=self.get_names_of_already_download_clips(
                path=output_path,
                prefix_len=YtUrlGetter.get_prefix_len(url=playlist_url),
            ),
            queue=online_clips_info,
        )

        # Create `downloaded_clips_info` queue
        downloaded_clips_info = Queue_mp()
        YtUrlGetter.put_downloaded_clip_info_to_queue(
            path=output_path,
            queue=downloaded_clips_info,
        )

        workers = []

        # create and start downloader workers
        print(f'Starting {self.num_of_downloader_workers} downloaders...')
        num_of_active_downloaders = Value('i', 0)
        for _ in range(self.num_of_downloader_workers):
            with num_of_active_downloaders.get_lock():
                num_of_active_downloaders.value += 1

            downloader = Thread(
                target=self.download_audio,
                kwargs={
                    'online_clips_info': online_clips_info,
                    'downloaded_clips_info': downloaded_clips_info,
                    'num_of_active_downloaders': num_of_active_downloaders,
                    'output_path': output_path,
                },
            )
            downloader.start()
            workers.append(downloader)

        # create and start converter workers
        print(f'Starting {os.cpu_count()} converters...')
        for _ in range(os.cpu_count()):
            converter = Process(
                target=self.convert_to_mp3_and_remove,
                kwargs={
                    'downloaded_clips_info': downloaded_clips_info,
                },
            )
            converter.start()
            workers.append(converter)

        [worker.join() for worker in workers]
        print(f'Playlist: {playlist_name} downloaded')
        print()

    def download_all_playlists(self):
        for playlist_url in self.playlist_urls:
            self.download_playlist(playlist_url)


if __name__ == '__main__':
    playlist_US17 = (
        'https://www.youtube.com/playlist?list='
        'PLda6VETsZ3sALHJVBmefYVvZXR9Gp-lqu'
    )
    playlist_multivariable_calculus = (
        'https://www.youtube.com/playlist?list='
        'PLSQl0a2vh4HC5feHa6Rc5c0wbRTx56nF7'
    )
    playlist_czytamy_nature = (
        'https://www.youtube.com/playlist?list='
        'PLuqpwpkBmbAn3lA7tY9lO3LaUOaRkAq1A'
    )

    yt_downloader = YtDownloader(
        urls=[
            playlist_US17,
            playlist_multivariable_calculus,
            playlist_czytamy_nature,
        ]
    )

    yt_downloader.download_all_playlists()
