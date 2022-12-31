import abc
import glob
import re
import pathlib
import pytube
import os.path
import moviepy.editor as editor

from itertools import compress
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass


@dataclass
class DownloadedParts:
    audio_fdir: str
    video_fdir: str | None


class YouTubeDownloaderHandler:
    def __init__(self):
        self.fdir_audio_urls = os.path.join(
            pathlib.Path.home(), r'Documents\YouTube_audio.txt'
        )
        self.fdir_video_urls = os.path.join(
            pathlib.Path.home(), r'Documents\YouTube_video.txt'
        )

        self.audio_urls = self.read_audio_urls()
        self.video_urls = self.read_video_urls()

    @staticmethod
    def read_urls(fdir_urls: str):
        with open(fdir_urls) as file:
            urls = file.readlines()
        return [url.rstrip() for url in urls]

    def read_audio_urls(self):
        return self.read_urls(fdir_urls=self.fdir_audio_urls)

    def read_video_urls(self):
        return self.read_urls(fdir_urls=self.fdir_video_urls)

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

    def get_clip_urls(self, urls: list[str]) -> list[str]:
        return [url for url in urls if self.is_video(url)]

    def get_channel_urls(self, urls: list[str]) -> list[str]:
        return [url for url in urls if self.is_channel(url)]

    def get_playlist_urls(self, urls: list[str]) -> list[str]:
        return [url for url in urls if self.is_playlist(url)]

    def go_audio(self) -> None:
        pass
        # clip_urls = self.get_clip_urls(urls=self.audio_urls)
        # channel_urls = self.get_channel_urls(urls=self.audio_urls)
        # playlist = self.get_playlist_urls(urls=self.audio_urls)
        #
        # YouTubeAudioClipsDownloader(urls=clip_urls).go()


class YouTubeDownloader(abc.ABC):
    @property
    @abc.abstractmethod
    def destination_folder(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def urls(self) -> list[str]:
        pass

    @property
    def num_of_online_videos(self) -> int:
        return len(self.urls)

    def all_fdirs_in_destination_folder(self) -> list[pathlib.Path]:
        return list(pathlib.Path(self.destination_folder).glob('*.*'))

    def all_fnames_in_destination_folder(self) -> list[str]:
        fdirs = self.all_fdirs_in_destination_folder()
        return [os.path.split(fdir)[1] for fdir in fdirs]

    def remove_bad_downloaded_files(self) -> None:
        for fdir in self.all_fdirs_in_destination_folder():
            if os.path.getsize(fdir) == 0:
                os.remove(fdir)

    @abc.abstractmethod
    def create_prefix(self, url: str) -> str:
        pass

    @staticmethod
    def pick_audio_stream(url: str) -> pytube.streams.Stream:
        audio_streams = pytube.YouTube(url).streams.filter(only_audio=True)
        return audio_streams.order_by('abr')[-1]

    @staticmethod
    def pick_video_stream(url: str) -> pytube.streams.Stream:
        resolutions = ['1080p', '720p', '480p', '360p', '240p', '144p']
        streams = pytube.YouTube(url).streams.filter(only_video=True)
        mp4_streams = streams.filter(mime_type='video/mp4')
        for resolution in resolutions:
            for stream in mp4_streams:
                if resolution in str(stream):
                    return stream

    @abc.abstractmethod
    def download(self, url: str, prefix: str) -> DownloadedParts:
        pass

    @abc.abstractmethod
    def convert(self, fdir: str) -> None:
        pass

    def is_downloaded(self, url: str) -> bool:
        fnames = self.all_fnames_in_destination_folder()
        fnames = [os.path.splitext(fname)[0] for fname in fnames]
        regex_pattern = re.compile(r'(?:\[\d+]|)(.+)')
        fnames = [
            regex_pattern.search(fname).group(1) for fname in fnames
        ]

        online_fname = pytube.YouTube(url).streams.first().default_filename
        online_fname = os.path.splitext(online_fname)[0]

        return online_fname in fnames

    def get_missing_urls(self, urls: list[str]) -> list[str]:
        print('Odfiltrowuję już pobrane pliki.')

        with ThreadPoolExecutor(max_workers=50) as executor:
            mask = list(executor.map(self.is_downloaded, urls))

        mask = [not x for x in mask]

        print(f'Znaleziono {sum(mask)} plików do pobrania.')

        return list(compress(urls, mask))

    @abc.abstractmethod
    def download_convert_delete(self, url: str) -> None:
        pass

    def download_convert_delete_all(self):
        urls = self.get_missing_urls(urls=self.urls)

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            executor.map(self.download_convert_delete, urls)

    def go(self) -> None:
        self.remove_bad_downloaded_files()
        self.download_convert_delete_all()


class YouTubeAudioDownloader(YouTubeDownloader, abc.ABC):
    def download(self, url: str, prefix: str) -> DownloadedParts:
        try:
            stream = self.pick_audio_stream(url)
            fdir = stream.download(
                output_path=self.destination_folder,
                filename_prefix=prefix
            )
        except KeyError:  # KeyError: 'streamingData'
            fdir = None  # don't download if can't

        return DownloadedParts(audio_fdir=fdir, video_fdir=None)

    def convert(self, components: DownloadedParts) -> None:
        if components.audio_fdir is not None:
            fdir = components.audio_fdir
            audio_clip = editor.AudioFileClip(fdir)
            audio_clip.write_audiofile(
                os.path.splitext(fdir)[0] + '.mp3', verbose=False, logger=None)
            audio_clip.close()

    def download_convert_delete(self, url: str) -> None:
        prefix = self.create_prefix(url)
        components = self.download(url, prefix)
        if not components.audio_fdir.endswith('.mp3'):
            self.convert(components)
            os.remove(components.audio_fdir)


class YouTubeVideoDownloader(YouTubeDownloader, abc.ABC):
    def download(self, url: str, prefix: str) -> DownloadedParts:
        try:
            audio_stream = self.pick_audio_stream(url)
            video_stream = self.pick_video_stream(url)

            audio_fdir = audio_stream.download(
                output_path=self.destination_folder,
                filename_prefix=prefix
            )
            video_fdir = video_stream.download(
                output_path=self.destination_folder,
                filename_prefix=prefix
            )
        except KeyError:  # KeyError: 'streamingData'
            audio_fdir, video_fdir = None, None  # don't download if can't

        return DownloadedParts(audio_fdir=audio_fdir, video_fdir=video_fdir)

    def convert(self, components: DownloadedParts) -> None:
        if (components.audio_fdir is not None and
                components.video_fdir is not None):

            audio_clip = editor.AudioFileClip(components.audio_fdir)
            video_clip = editor.VideoFileClip(components.video_fdir)

            new_audio_clip = editor.CompositeAudioClip([audio_clip])
            video_clip.audio = new_audio_clip
            video_clip.write_videofile(
                os.path.splitext(components.video_fdir)[0] + '100.mp4',
                verbose=False, logger=None)

            audio_clip.close()
            video_clip.close()
            new_audio_clip.close()

    def download_convert_delete(self, url: str) -> None:
        prefix = self.create_prefix(url)
        components = self.download(url, prefix)
        self.convert(components)
        # os.remove(components.audio_fdir)


class YouTubeMixDownloader(YouTubeDownloader, abc.ABC):
    def __init__(self, urls: list[str]):
        self._urls = urls

    @property
    def urls(self):
        return self._urls

    def create_prefix(self, url: str) -> str:
        return ''


class YouTubeOrderedDownloader(YouTubeDownloader, abc.ABC):
    def __init__(self, urls: list[str]):
        self._urls = urls

    @property
    def urls(self) -> list[str]:
        return self._urls

    def create_prefix(self, url: str) -> str:  # '[001]', '[002]', ...
        vid_num = self.urls.index(url)
        num_of_digits = len(str(self.num_of_online_videos))
        return f"[{str(vid_num + 1).zfill(num_of_digits)}]"


class YouTubeAudioClipsDownloader(YouTubeAudioDownloader,
                                  YouTubeMixDownloader):
    def __init__(self, urls: list[str]):
        YouTubeAudioDownloader.__init__(self)
        YouTubeMixDownloader.__init__(self, urls)

    @property
    def destination_folder(self) -> str:
        return os.path.join(
            pathlib.Path.home(), r'Downloads\YouTube\audio\mix'
        )


class YouTubeVideoClipsDownloader(YouTubeVideoDownloader,
                                  YouTubeMixDownloader):
    def __init__(self, urls: list[str]):
        YouTubeVideoDownloader.__init__(self)
        YouTubeMixDownloader.__init__(self, urls)

    @property
    def destination_folder(self) -> str:
        return os.path.join(
            pathlib.Path.home(), r'Downloads\YouTube\video\mix'
        )


class YouTubeAudioChannelDownloader(YouTubeAudioDownloader,
                                    YouTubeOrderedDownloader):

    def __init__(self, channel_url: str):
        self.yt_channel = pytube.Channel(channel_url)

        YouTubeAudioDownloader.__init__(self)
        YouTubeOrderedDownloader.__init__(
            self, urls=list(reversed(self.yt_channel.video_urls)))

    @property
    def destination_folder(self) -> str:
        return os.path.join(
            pathlib.Path.home(),
            r'Downloads\YouTube\audio',
            self.yt_channel.channel_name
        )


class YouTubeAudioPlaylistDownloader(YouTubeAudioDownloader,
                                     YouTubeOrderedDownloader):

    def __init__(self, playlist_url: str):
        self.yt_playlist = pytube.Playlist(playlist_url)

        YouTubeAudioDownloader.__init__(self)
        YouTubeOrderedDownloader.__init__(
            self, urls=list(reversed(self.yt_playlist.video_urls)))

    @property
    def destination_folder(self) -> str:
        return os.path.join(
            pathlib.Path.home(),
            r'Downloads\YouTube\audio',
            self.yt_playlist.title
        )


class YtClipsDownloader:
    def __init__(self, urls: list[str]):
        self.urls = urls
        self.num_of_urls = len(self.urls)

        self.destination_folder = (
                str(pathlib.Path.home()) +
                str(pathlib.Path(r'//Downloads//YouTube//')))

    @staticmethod
    def get_best_audio_stream(url: str) -> pytube.streams.Stream:
        """
        Gets `url` to video on YT and return `pytube.streams.Stream` with
        the best audio sampling rate.

        Audio sampling rate is given by `abr` attribute.
        """
        audio_streams = pytube.YouTube(url).streams.filter(only_audio=True)
        return audio_streams.order_by('abr')[-1]

    def all_fdirs_in_dir(self) -> list[str]:
        directory = self.destination_folder
        if not pathlib.Path(directory).exists():
            return []

        cwd = os.getcwd()
        os.chdir(directory)
        fdirs = [directory + '//' + fname for fname in glob.glob("*.*")]
        os.chdir(cwd)

        fdirs.sort()
        return fdirs

    def download_file(self, url: str, video_num: int) -> str:
        print(f"Downloading file {video_num + 1}"
              f" out of {self.num_of_urls} ...")
        try:
            stream = self.get_best_audio_stream(url)
            fdir = stream.download(output_path=self.destination_folder)
        except KeyError:  # KeyError: 'streamingData'
            fdir = None  # don't download if can't

        return fdir

    def convert_to_mp3(self, fdir: str, video_num: int) -> None:
        print(f"Clip {video_num + 1} out of {self.num_of_urls}"
              f" is converted to mp3 ...")

        audio_clip = editor.AudioFileClip(fdir)
        audio_clip.write_audiofile(
            os.path.splitext(fdir)[0] + '.mp3', verbose=False, logger=None)
        audio_clip.close()

    def download_covert_delete(self, url: str, video_num: int):
        fdir = self.download_file(url, video_num)
        self.convert_to_mp3(fdir, video_num)
        os.remove(fdir)

    def is_downloaded(self, url: str) -> bool:
        fdirs = self.all_fdirs_in_dir()
        online_fname = self.get_best_audio_stream(url).default_filename
        return any(os.path.splitext(online_fname)[0] in fdir for fdir in fdirs)

    def get_missing_urls(self) -> list[str]:
        with ThreadPoolExecutor(max_workers=50) as executor:
            mask = list(executor.map(self.is_downloaded, self.urls))

        mask = [not x for x in mask]
        return list(compress(self.urls, mask))

    def download_convert_delete_all(self):
        urls = self.get_missing_urls()

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            executor.map(self.download_covert_delete, urls, range(len(urls)))
        print("Pobieranie ukończone pomyślnie, program kończy działanie.")

    def remove_bad_downloaded_files(self) -> None:
        for fdir in self.all_fdirs_in_dir():
            if os.path.getsize(fdir) == 0:
                os.remove(fdir)

    def go(self):
        self.remove_bad_downloaded_files()
        self.download_convert_delete_all()


if __name__ == "__main__":
    clip_urls = [
        'https://www.youtube.com/watch?v=WAvT_w5NELQ',
        'https://www.youtube.com/watch?v=4A71Wswcy6o',
        'https://www.youtube.com/watch?v=xbeM2I0Cidw',
        'https://www.youtube.com/watch?v=bsWTCBl2-Fk&list=RDMM&index=3',
        'https://www.youtube.com/watch?v=l_AS1FWcLYI&list=RDMM&index=8',
    ]
    c_urls = ['https://www.youtube.com/watch?v=me7yVknCSFw']
    # YouTubeAudioClipsDownloader(clip_urls).go()
    YouTubeVideoClipsDownloader(urls=c_urls).go()

    chan_url = ('https://www.youtube.com/channel/'
                'UClWdNeoEgVF_lWRWdDpa8iw/videos')
    # YouTubeAudioChannelDownloader(channel_url=chan_url).go()

    play_url = ('https://www.youtube.com/playlist?'
                'list=PL9_eqkWy-KHRyfQZppT1skwefuohji9Kr')
    # YouTubeAudioPlaylistDownloader(playlist_url=play_url).go()

    # YtClipsDownloader(urls=[
    #     'https://www.youtube.com/watch?v=WAvT_w5NELQ',
    #     'https://www.youtube.com/watch?v=4A71Wswcy6o',
    #     'https://www.youtube.com/watch?v=xbeM2I0Cidw',
    # ]).go()
