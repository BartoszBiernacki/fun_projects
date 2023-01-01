import os
import pathlib
import re
import pytube
from multiprocessing import Process, Queue, current_process
from threading import Thread
import moviepy.editor as editor
import glob
from bs4 import BeautifulSoup
import requests
from concurrent.futures import ThreadPoolExecutor


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
    def get_clip_urls_from_playlist_url(playlist_url: str) -> list[str]:
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
    def get_video_titles_from_playlist(cls, url: str, num_workers=32) -> list[str]:
        executor = ThreadPoolExecutor(max_workers=num_workers)
        urls = pytube.Playlist(url).video_urls
        titles = list(executor.map(cls.get_video_title, urls))
        return titles

    @staticmethod
    def remove_illegal_characters(filename: str) -> str:
        illegal_characters = r'[\\/:*?"<>|]'
        return re.sub(illegal_characters, '', filename)

    @classmethod
    def get_default_filenames(cls, titles: list[str]) -> list[str]:
        return [cls.remove_illegal_characters(title) for title in titles]

    @staticmethod
    def get_prefix_len(url: str) -> int:
        if YtUrlTypeRecognizer.get_playlist_urls([url]):
            return len(str(pytube.Playlist(url).length)) + 2
        elif YtUrlTypeRecognizer.get_channel_urls([url]):
            return len(str(pytube.Channel(url).length)) + 2
        else:
            return 0


class YtDownloader:
    def __init__(self, urls: list[str]):
        
        self.playlist_urls = YtUrlTypeRecognizer.get_playlist_urls(urls)
        self.channel_urls = YtUrlTypeRecognizer.get_channel_urls(urls)
        self.clip_urls = YtUrlTypeRecognizer.get_clip_urls(urls)

        self.output_base_path = os.path.join(self.get_download_folder(), r'YouTube2\audio')
        self.num_of_downloader_workers = 10

    @staticmethod
    def get_download_folder():
        # Expand the '~' symbol to the path of the user's home directory
        home_dir = os.path.expanduser('~')
        # Return the default download folder path
        return os.path.join(home_dir, 'Downloads')

    @staticmethod
    def create_prefix(url_idx: int, num_of_all_urls: int) -> str:
        """ Returns prefix (str) like `[001]` `[002]` and so on.

        Useful for sorting downloaded files by order of their appearance on
        YouTube.
        """
        num_of_digits = len(str(num_of_all_urls))
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

    def put_missing_urls_playlist_to_queue(
            self,
            playlist_url: str,
            queue_url: Queue,
            queue_prefix: Queue) -> None:
        print('Putting missing urls to queue')

        print('Getting names of downloaded clips...')
        downloaded_fnames = self.get_names_of_already_download_clips(
            path=self.get_output_path(url=playlist_url),
            prefix_len=YtUrlGetter.get_prefix_len(url=playlist_url)
        )
        print('downloaded_fnames:')
        list(map(print, downloaded_fnames))
        print()

        print('Getting names of online clips...')
        online_titles = YtUrlGetter.get_video_titles_from_playlist(playlist_url)
        online_names = YtUrlGetter.get_default_filenames(titles=online_titles)
        print('online_names:')
        list(map(print, online_names))
        print()

        urls = YtUrlGetter.get_clip_urls_from_playlist_url(playlist_url)
        for idx, (url, online_name) in enumerate(zip(urls, online_names)):
            if online_name not in downloaded_fnames:
                queue_url.put(url)
                queue_prefix.put(self.create_prefix(idx, len(urls)))

    @staticmethod
    def put_not_converted_clips_to_queue(path: str, queue: Queue) -> None:
        print('Putting not converted urls to queue')
        if os.path.exists(path):
            # search all files not in mp3 format
            [queue.put(fdir) for fdir in glob.glob(path + r'\*.[!mp3]')]

    @staticmethod
    def download_audio(
            url_queue: Queue,
            output_path: str,
            prefix_queue: Queue,
            out_queue: Queue) -> None:

        while not url_queue.empty():
            url = url_queue.get()
            prefix = prefix_queue.get()

            audio_stream = YtUrlGetter.get_audio_stream(url)
            print(f"[downloading] {prefix}"
                  f" {audio_stream.default_filename}", flush=True)
            out_queue.put(audio_stream.download(
                output_path=output_path, filename_prefix=prefix))

        out_queue.put('')

    @staticmethod
    def convert_to_mp3_and_remove(queue_fdir: Queue) -> None:
        while True:
            fdir = queue_fdir.get()
            if fdir != '':
                print(f"[mp3 converting] {os.path.split(fdir)[1]}", flush=True)
                audio_clip = editor.AudioFileClip(fdir)
                audio_clip.write_audiofile(
                    os.path.splitext(fdir)[0] + '.mp3',
                    verbose=False,
                    logger=None,
                )
                audio_clip.close()
                os.remove(fdir)
            else:
                queue_fdir.put('')
                break

    def download_playlist(self, playlist_url: str) -> None:
        fdir_queue, url_queue, prefix_queue = Queue(), Queue(), Queue()
        output_path = self.get_output_path(playlist_url)

        # prepare queues
        self.put_not_converted_clips_to_queue(output_path, fdir_queue)
        self.put_missing_urls_playlist_to_queue(
            playlist_url, url_queue, prefix_queue)

        # keep all created processes to join them
        processes = []

        # create and start downloader workers
        for _ in range(self.num_of_downloader_workers):
            downloader = Thread(
                target=self.download_audio,
                args=(url_queue, output_path, prefix_queue, fdir_queue)
            )
            downloader.start()
            processes.append(downloader)

        # create and start converter workers
        for _ in range(os.cpu_count()):
            converter = Process(
                target=self.convert_to_mp3_and_remove,
                args=(fdir_queue,))
            converter.start()
            processes.append(converter)

        [process.join() for process in processes]

    def download_all_playlists(self):
        for playlist_url in self.playlist_urls:
            self.download_playlist(playlist_url)


if __name__ == '__main__':
    yt_downloader = YtDownloader(
        ['https://www.youtube.com/playlist?list=PLda6VETsZ3sALHJVBmefYVvZXR9Gp-lqu']
    )

    yt_downloader.download_all_playlists()

        