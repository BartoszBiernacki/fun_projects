"""
NOTES:
    *   changed `channel_name` function in `extract.py` to handle new
    YouTube channel url format: https://youtube.com/@{channel_name}/*`
    details -- https://github.com/pytube/pytube/issues/1443

    * changed `channel.py` to Fix a bug that caused the channel's
     video list to be empty.
    details -- https://github.com/pytube/pytube/pull/1409
    detail2 -- https://github.com/pytube/pytube/pull/1409/files/42a7d8322dd7749a9e950baf6860d115bbeaedfc
"""
from __future__ import annotations

import os
import time
import random
import re
import logging
import pytube
import pytube.exceptions
import moviepy.editor as editor
import glob
import requests
import http.client

from dataclasses import dataclass
from multiprocessing import Value, Process
from multiprocessing import Queue as Queue_mp  # Queue for multiprocessing
from queue import Queue as Queue_mth  # Queue for multithreading
from threading import Thread
from enum import Enum, auto
from typing import Optional
from config import Config
from bs4 import BeautifulSoup

from utils import StringUtils


class ContainerType(Enum):
    """Enum representing `container` type."""
    CLIP = auto()
    PLAYLIST = auto()
    CHANNEL = auto()


@dataclass
class UrlAndIdx:
    """Simple data class storing clip url and its index if url is element of
    channel or playlist.

    Class instances are used as elements of multithreading queue to scrape
    additional data like video title.
    Scraped data are used to create `OnlineClipInfo` objects."""
    url: str
    idx: Optional[int] = None


@dataclass
class OnlineClipInfo:
    """Simple data class storing some information about online clips.

    Class instances are generated during initial scraping, and they are
    putted into container's queue_mp to be downloaded later."""
    url: str
    prefix: str = None
    default_filename: str = None
    is_downloaded: bool = None
    download_attempts: int = 0


@dataclass
class DownloadedClipInfo:
    """Simple data class storing some information about downloaded clips.

    Class instances are generated in two scenarios:
        * before (parallely) scraping online clips, using only already
        downloaded files. Useful if when starting downloader there are some
        downloaded files but for some reason not converted.
        * after download each clip.

    Class instances are always putted into container multiprocessing queue
    and will be converted to `CONFIG._DESIRED_EXTENSIONS` later."""

    file_directory: str = None
    filename: str = None
    is_converted: bool = None

    def __post_init__(self):
        if self.file_directory is not None:
            if not self.filename:
                self.filename = os.path.split(self.file_directory)[1]
            if self.is_converted is None:
                self.is_converted = self._is_converted(self.file_directory)

    @staticmethod
    def _is_converted(fdir: str):
        return os.path.splitext(fdir)[1] in Config.DESIRED_EXTENSIONS_DOTS


@dataclass
class Container:
    """Container stores all information about all clips:
        * on single playlist or
        * on single channel or
        * which are not belong to any playlist or channel.

    Any high level function should get container as it base argument and
    do not ask for any additional YouTube-related information.
    """
    type: ContainerType
    name: str
    urls: list[str]
    output_dir: str = None
    url_idx_queue: Queue_mth = None
    online_clips_info_queue: Queue_mth = None
    downloaded_clips_info_queue: Queue_mp = None

    def __post_init__(self):
        self.output_dir = os.path.join(Config.BASE_OUTPUT_PATH, self.name)

        # Note: queues must be created here, not in constructor.
        #       Otherwise, all instances of Container share the same queue!
        self.url_idx_queue = Queue_mth()
        self.online_clips_info_queue = Queue_mth()
        self.downloaded_clips_info_queue = Queue_mp()

        ContainerUtils.remove_corrupted_files(container=self)
        ContainerUtils.remove_converted_files(container=self)
        ContainerUtils.remove_duplicated_files(container=self)
        ContainerUtils.put_url_idx_to_queue(container=self)
        ContainerUtils.put_online_clips_info_to_queue(container=self)
        ContainerUtils.put_downloaded_clips_info_to_queue(container=self)


class ContainerUtils:
    """This class is responsible for easy Container-related operations,
    which should be done, right after `Container` creation."""

    @staticmethod
    def remove_corrupted_files(container: Container):
        """Remove corrupted files assuming their size is 0."""

        to_delete = [fdir
                     for fdir in ContainerUtils.downloaded_fdirs(container)
                     if os.path.getsize(fdir) == 0]

        logging.info(f'Deleting [{len(to_delete)}] corrupted '
                     f'files in {container.name}...')

        list(map(os.remove, to_delete))

    @staticmethod
    def remove_converted_files(container: Container):
        """Assume we have the following files:
        * `clip.mp3`
        * `clip.mp4`
        * `clip.webem`
        * `other_clip1.webem`
        * `other_clip2.mp3`
        and `Config._DESIRED_EXTENSIONS = ['mp3', 'mp4']`
        then:
        - remove `[clip.mp4, clip.webem]`
        + keep `[clip.mp3, other_clip1.webem, other_clip2.mp3]`
        """

        # Guard statement
        if not Config.DELETE_AFTER_CONVERSION:
            return

        fdirs = ContainerUtils.downloaded_fdirs(container)
        unique = {}
        for fdir in fdirs:
            if (
                    ((fname := StringUtils.get_fname_without_extension(fdir)) not in unique)
                    and (os.path.splitext(fdir)[1] in Config.DESIRED_EXTENSIONS_DOTS)
            ):
                unique[fname] = fdir

        to_delete = [fdir for fdir in fdirs if (
                    (StringUtils.get_fname_without_extension(fdir) in unique)
                    and (fdir not in unique.values())
            )]
        logging.info(f'Deleting [{len(to_delete)}] already converted '
                     f'files in {container.name}...')

        list(map(os.remove, to_delete))

    @classmethod
    def remove_duplicated_files(cls, container: Container) -> None:
        """Remove duplicates from disk.

         Duplicates are files, which differ by:
            * prefix or
            * number of dots in names (keep name with more dots).

        Duplicates have might occur while using intermediate form of
        this program, so I decided to add this functionality."""

        unique = {}
        for fdir in cls.downloaded_fdirs(container):
            no_prefix = StringUtils.get_fname_without_prefix(fdir)
            no_dots = no_prefix.replace('.', '')
            if no_dots not in unique:
                unique[no_dots] = fdir
            else:
                previous_dir = unique[no_dots]
                if len(previous_dir) >= len(fdir):
                    os.remove(fdir)
                else:
                    unique[no_dots] = fdir
                    os.remove(previous_dir)

    @staticmethod
    def put_url_idx_to_queue(container: Container) -> None:
        """Puts `UrlAndIdx` instances to container's queue."""
        if container.type == ContainerType.CHANNEL:
            container.urls = list(reversed(container.urls))

        for idx, url in enumerate(container.urls):
            if container.type != ContainerType.CLIP:
                container.url_idx_queue.put(UrlAndIdx(url=url, idx=idx))
            else:
                container.url_idx_queue.put(UrlAndIdx(url=url))

    @staticmethod
    def downloaded_fdirs(container: Container) -> list[str]:
        """Return all fdirs in `Container` download directory."""

        if os.path.exists(container.output_dir):
            return glob.glob(os.path.join(container.output_dir, r'*.*'))
        else:
            return []

    @classmethod
    def downloaded_fnames_no_ext(cls, container: Container) -> list[str]:
        """Return all fnames in `Container` download directory
        without filetype extension.

        `C:User\\...\\[004]name.mp3` --> `[004]name`."""

        return [StringUtils.get_fname_without_extension(fdir)
                for fdir in cls.downloaded_fdirs(container)]

    @classmethod
    def downloaded_fnames_no_prefix_no_ext(
            cls, container: Container) -> list[str]:
        """Return all fnames in `Container` download directory
        without prefix and filetype extension.

        `C:User\\...\\[004]name.mp3` --> `name`."""

        return [StringUtils.get_fname_without_prefix_and_extension(fdir)
                for fdir in cls.downloaded_fdirs(container)]

    @classmethod
    def put_online_clips_info_to_queue(cls, container: Container) -> None:
        """Fill `container.online_clips_info_queue`.

        This function using multithreading to minimize scraping time.
        Workers come from `YtScraper` class.

        Threads are NOT JOINED, because there is no need for them to be.
        That is because other functions, which depends on this queue waits for
        `ONLINE_INFO_SENTINEL` which is putted after getting and handling
        last item from `container.url_idx_queue`. This solution allows
        for scraping, downloading and converting at the same time.
        """

        logging.info(f'Putting OnlineClipInfo of {container.name} to queue...')

        # keep num of all working scrapers to know when to put stop sentinel
        # for downloaders
        num_of_active_clip_info_putters = Value('i', 0)

        # extract downloaded fnames without prefix and extension (only once)
        downloaded_fnames_no_prefix_no_ext = \
            cls.downloaded_fnames_no_prefix_no_ext(container)

        # create scraper workers
        for _ in range(Config.NUM_SCRAPERS):
            with num_of_active_clip_info_putters.get_lock():
                num_of_active_clip_info_putters.value += 1

            scraper = Thread(
                target=YtScraper.put_online_clips_to_info_queue,
                kwargs={
                    'container': container,
                    'downloaded_fnames_no_prefix_no_ext': downloaded_fnames_no_prefix_no_ext,
                    'num_of_active_clip_info_putters': num_of_active_clip_info_putters,
                }
            )
            scraper.start()

    @classmethod
    def put_downloaded_clips_info_to_queue(cls, container: Container) -> None:
        """Fill `container.downloaded_clips_info_queue` with clips downloaded
        before `YtDownloader.handle_container` was run.
        Those clips may be result of previous downloading.

        This function does not depend on internet connection, but only on
        filenames of existing files, which make it very fast, so
        multithreading is no needed."""

        for fdir in cls.downloaded_fdirs(container):
            container.downloaded_clips_info_queue.put(
                DownloadedClipInfo(file_directory=fdir))


class ContainerCreator:
    """This class has only one important function
    `create_containers_from_urls` which is responsible for automatic creation
    of containers based on provided urls.

    If divides urls into one of three categories:
        * clip urls
        * playlist urls
        * channel urls
    Then for each url of type playlist or channel one container is created,
    and there is only container for all the clip urls."""

    @staticmethod
    def _is_clip(url: str) -> bool:
        """Is the url an url to clip?"""

        video_pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
        return bool(re.search(video_pattern, url))

    @staticmethod
    def _is_channel(url: str) -> bool:
        """Is the url an url to channel?"""

        channel_patterns = [
            r"(?:\/(c)\/([%\d\w_\-]+)(\/.*)?)",
            r"(?:\/(channel)\/([%\w\d_\-]+)(\/.*)?)",
            r"(?:\/(u)\/([%\d\w_\-]+)(\/.*)?)",
            r"(?:\/(user)\/([%\w\d_\-]+)(\/.*)?)",
            r"(?:\/(@)([%\d\w_\-]+)(\/.*)?)",
        ]
        return any(bool(re.search(pattern, url))
                   for pattern in channel_patterns)

    @staticmethod
    def _is_playlist(url: str) -> bool:
        """Is the url an url to playlist?"""

        playlist_pattern = r'www\.youtube\.com/playlist\?list='
        return bool(re.search(playlist_pattern, url))

    @classmethod
    def _filter_for_playlist_urls(cls, urls: list[str]):
        """Returns urls of playlists."""
        return [url for url in urls if cls._is_playlist(url)]

    @classmethod
    def _filter_for_channel_urls(cls, urls: list[str]):
        """Returns urls of channels."""
        return [url for url in urls if cls._is_channel(url)]

    @classmethod
    def _filter_for_urls_of_single_clips(cls, urls: list[str]):
        """Returns urls of single clips."""
        return [url for url in urls if cls._is_clip(url)]

    @classmethod
    def create_containers_from_urls(cls, urls: list[str]) -> list[Container]:
        """Get all urls provided by user, group them as single clips,
        playlists and channels. Then create and return corresponding
        containers."""

        containers = []

        for channel_url in cls._filter_for_channel_urls(urls):
            channel = pytube.Channel(channel_url)
            print(f'{channel = }')
            containers.append(
                Container(
                    type=ContainerType.CHANNEL,
                    name=StringUtils.remove_os_illegal_characters(
                        channel.channel_name),
                    urls=list(channel.video_urls),
                )
            )

        for playlist_url in cls._filter_for_playlist_urls(urls):
            playlist = pytube.Playlist(playlist_url)
            containers.append(
                Container(
                    type=ContainerType.PLAYLIST,
                    name=StringUtils.remove_os_illegal_characters(
                        playlist.title),
                    urls=list(playlist.video_urls),
                )
            )

        if clip_urls := cls._filter_for_urls_of_single_clips(urls):
            containers.append(
                Container(
                    type=ContainerType.CLIP,
                    name='clips',
                    urls=clip_urls,
                )
            )

        return containers


class YtScraper:
    """Purpose of this class is to provide information about
     online clips before downloading them."""

    @staticmethod
    def _get_video_title(url: str) -> str:
        """Returns video title. Try to get it 10 times in order to be
        safe about some network issues. Be aware that title can
        contain illegal os characters."""
        for _ in range(10):
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                title_element = soup.find('title')
                title = title_element.text

                # Return the title without at the end ' - youtube'
                return title[:-10]

            except ConnectionError:
                time.sleep(random.random())

    @staticmethod
    def _get_playlist_name(url: str) -> str:
        """Returns playlist title without illegal os characters."""
        name = pytube.Playlist(url).title
        return StringUtils.remove_os_illegal_characters(name)

    @staticmethod
    def _get_channel_name(url: str) -> str:
        """Returns channel title without illegal os characters."""
        name = pytube.Channel(url).channel_name
        return StringUtils.remove_os_illegal_characters(name)

    @classmethod
    def _get_video_name(cls, url: str) -> str:
        """Returns video title without illegal os characters."""
        title = cls._get_video_title(url)
        return StringUtils.remove_os_illegal_characters(title)

    @staticmethod
    def _construct_prefix(
            idx_of_url: int,
            n_urls_in_container: int,
            container_type: ContainerType,
    ) -> str:
        """ Returns prefix (str) of filename like:
         `[001]`, `[002]`, ..., `[065]`.

        Useful for sorting downloaded files in file explorer
        by order of their appearance on YouTube."""

        if container_type == ContainerType.CLIP:
            return ''

        n_digits = len(str(n_urls_in_container))
        # start at 1 not 0, and last prefix should be [0xx] not [xx]
        return f"[{str(idx_of_url + 1).zfill(n_digits + 1)}]"

    @classmethod
    def put_online_clips_to_info_queue(
            cls,
            container: Container,
            downloaded_fnames_no_prefix_no_ext: list[str],
            num_of_active_clip_info_putters: Value('i'),
    ) -> None:
        """Use this function by workers to speed up filling
        `container.online_clips_info_queue` with clips that needs to
        be downloaded.

        url_queue --> online_clips_info_queue
        """

        queue_in = container.url_idx_queue
        queue_out = container.online_clips_info_queue

        while not queue_in.empty():
            item = queue_in.get()
            url, idx = item.url, item.idx

            # Put clip to queue only if clip's data were scraped properly
            if default_filename := cls._get_video_name(url):
                if default_filename not in downloaded_fnames_no_prefix_no_ext:

                    queue_out.put(
                        OnlineClipInfo(
                            url=url,
                            prefix=cls._construct_prefix(
                                idx_of_url=idx,
                                n_urls_in_container=len(container.urls),
                                container_type=container.type
                            ),
                            default_filename=default_filename,
                            is_downloaded=False,
                        )
                    )

        # If in_queue is empty and currently working putter is the last one
        # put ONLINE_INFO_SENTINEL at the end of queue_out which will be
        # stop flag for downloaders.
        with num_of_active_clip_info_putters.get_lock():
            if num_of_active_clip_info_putters.value == 1:
                queue_out.put(
                    YtDownloader.ONLINE_INFO_SENTINEL
                )

                logging.info(
                    f'[{container.online_clips_info_queue.qsize()} '
                    f'clips need to be downloaded]'
                )

            else:
                num_of_active_clip_info_putters.value -= 1

    @staticmethod
    def get_audio_stream(clip_url: str) -> pytube.streams.Stream | None:
        """Get an audio stream from YT clip.
        Firstly try to get stream with desired filetypes and
        quality. If desired filetype is not available return
        stream with different filetype and use other functions
        to convert it."""

        def get_stream_idx(_streams: pytube.query.StreamQuery) -> int:
            if Config.QUALITY < 0:
                return 0
            elif Config.QUALITY > 1:
                return 1
            else:
                return round(Config.QUALITY * (len(_streams) - 1))

        try:
            audio_streams = pytube.YouTube(
                url=clip_url,
                use_oauth=True,
                allow_oauth_cache=True
            ).streams.filter(only_audio=True).order_by('filesize')

            # return stream with desired extension and quality
            for ext in Config.DESIRED_EXTENSIONS_NO_DOTS:
                if streams := audio_streams.filter(subtype=ext):
                    stream_idx = get_stream_idx(streams)
                    return streams[stream_idx]

            # if there is no matching on filetype just
            # return stream with desired quality
            if audio_streams:
                stream_idx = get_stream_idx(audio_streams)
                return audio_streams[stream_idx]

        except pytube.exceptions.VideoUnavailable:
            logging.warning(f'[Stream not available] {clip_url}')
            return None


class YtDownloader:
    """This class is the main dish. It set up all container,
     and manages all downloader and converter workers.

    To download something from YT (clip, channel, playlist) just create
    instance of this class by providing all urls at once in list.
    """

    ONLINE_INFO_SENTINEL = OnlineClipInfo(url='')
    DOWNLOADED_INFO_SENTINEL = DownloadedClipInfo()

    def __init__(self, urls: list[str]):

        self.containers = ContainerCreator.create_containers_from_urls(urls)
        self.handle_all_containers()

    @classmethod
    def audio_downloader(
            cls,
            container: Container,
            num_of_active_downloaders: Value('i'),
    ) -> None:
        """Use this function to download all clips from container.
        Run this function as a worker in multiple threads to
        speed up download time.

        online_clips_info_queue --> downloaded_clips_info_queue
        """

        queue_in = container.online_clips_info_queue
        queue_out = container.downloaded_clips_info_queue

        # Not just until queue_in is empty, because
        # clip_info_putter may be slower than downloader
        while True:
            online_clip_info = queue_in.get()

            if online_clip_info == cls.ONLINE_INFO_SENTINEL:
                queue_in.put(cls.ONLINE_INFO_SENTINEL)
                break

            elif online_clip_info.is_downloaded:
                logging.info(
                    f'[download skipping exist] '
                    f'{online_clip_info.default_filename}'
                )

            elif online_clip_info.download_attempts <= 10:
                audio_stream = YtScraper.get_audio_stream(online_clip_info.url)
                if audio_stream is not None:
                    ext = os.path.splitext(audio_stream.default_filename)[1]
                    logging.info(f"[downloading] {online_clip_info.prefix}"
                                 f" {online_clip_info.default_filename + ext}")

                    try:
                        fdir = audio_stream.download(
                            output_path=container.output_dir,
                            filename=online_clip_info.default_filename + ext,
                            filename_prefix=online_clip_info.prefix
                        )
                        queue_out.put(DownloadedClipInfo(file_directory=fdir))

                    except (http.client.IncompleteRead, TimeoutError):
                        # Try again later
                        logging.warning(
                            f'[download '
                            f'{online_clip_info.download_attempts + 1} '
                            f'attempt unsuccessful] '
                            f'{online_clip_info.default_filename}'
                        )
                        online_clip_info.download_attempts += 1
                        queue_in.put(online_clip_info)
            else:
                logging.error(
                    f'[download error] '
                    f'{online_clip_info.default_filename} '
                    f'can not be downloaded after '
                    f'{online_clip_info.download_attempts} '
                    f'attempts and will be skipped!'
                )

        with num_of_active_downloaders.get_lock():
            if num_of_active_downloaders.value == 1:
                queue_out.put(cls.DOWNLOADED_INFO_SENTINEL)

            num_of_active_downloaders.value -= 1

    @classmethod
    def audio_converter(
            cls,
            downloaded_clips_info: Queue_mp[DownloadedClipInfo],
    ) -> None:
        """Use this function to convert all downloaded clips from
         container's `downloaded_clips_info_queue`.

        Run this function as a worker in multiple process to
        speed up conversion time.

        Please note, that this function, (probably) can't be rewritten to
        accept just container, because it meant to be run by different
        processes, so argument must be `multiprocessing.Queue`."""

        logging.basicConfig(level=logging.INFO)

        def is_or_was_converted(file_path) -> bool:
            """Returns True if folder contain file with the same name
            as given and with filetype, which is considered as converted."""

            folder, fname = os.path.split(file_path)
            fname_no_ext, ext = os.path.splitext(fname)

            for fdir in glob.glob(os.path.join(folder, '*.*')):
                other_fname = os.path.split(fdir)[1]
                other_fname_no_ext, other_ext = os.path.splitext(other_fname)
                if other_fname_no_ext == fname_no_ext:
                    if other_ext in Config.DESIRED_EXTENSIONS_DOTS:
                        return True
            return False

        while True:
            clip_info = downloaded_clips_info.get()

            if clip_info == cls.DOWNLOADED_INFO_SENTINEL:
                break

            if not is_or_was_converted(file_path=clip_info.file_directory):
                logging.info(f"[converting] {clip_info.filename}")
                audio_clip = editor.AudioFileClip(clip_info.file_directory)

                audio_clip.write_audiofile(
                    (os.path.splitext(clip_info.file_directory)[0]
                     + Config.DESIRED_EXTENSIONS_DOTS[0]),
                    verbose=False,
                    logger=None,
                )
                audio_clip.close()

        downloaded_clips_info.put(cls.DOWNLOADED_INFO_SENTINEL)

    def run_downloaders_for_container(
            self,
            container: Container,
            workers: list[Thread | Process]
    ) -> None:
        """Create and run downloader workers.

        Downloaders are added to list of workers in order to join them later
        with other workers"""

        logging.info(f'[{container.name} have {len(container.urls)} clips]')
        num_of_active_downloaders = Value('i', 0)
        for _ in range(Config.NUM_DOWNLOADERS):
            with num_of_active_downloaders.get_lock():
                num_of_active_downloaders.value += 1

            downloader = Thread(
                target=self.audio_downloader,
                kwargs={
                    'container': container,
                    'num_of_active_downloaders': num_of_active_downloaders,
                },
            )
            downloader.start()
            workers.append(downloader)

    def run_converters_for_container(
            self,
            container: Container,
            workers: list[Thread | Process],
    ) -> None:
        """Create and run converter workers.

        Converters are added to list of workers in  order to join them later
        with other workers."""

        logging.info(
            f'Starting {Config.NUM_CONVERTERS} converters '
            f'for {container.name}...'
        )

        for _ in range(Config.NUM_CONVERTERS):

            converter = Process(
                target=self.audio_converter,
                kwargs={
                    'downloaded_clips_info': container.downloaded_clips_info_queue,
                },
            )
            converter.start()
            workers.append(converter)

    def handle_container(self, container: Container) -> None:
        """Download and convert container."""
        logging.info(f'Handling container: {container.name}')

        workers = []
        self.run_downloaders_for_container(
            container=container, workers=workers)
        self.run_converters_for_container(
            container=container, workers=workers)

        [worker.join() for worker in workers]
        ContainerUtils.remove_converted_files(container)
        logging.info(f'Container: {container.name} handled')

    def handle_all_containers(self):
        """Download and convert all containers."""
        for container in self.containers:
            self.handle_container(container)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Playlists
    US17 = 'https://www.youtube.com/playlist?list=PLda6VETsZ3sALHJVBmefYVvZXR9Gp-lqu'
    multivariable_calculus = 'https://www.youtube.com/playlist?list=PLSQl0a2vh4HC5feHa6Rc5c0wbRTx56nF7'
    czytamy_nature = 'https://www.youtube.com/playlist?list=PLuqpwpkBmbAn3lA7tY9lO3LaUOaRkAq1A'
    szkolna_funny = 'https://www.youtube.com/playlist?list=PLnD8c5cfKMwQRwUPMBg_fvFuxV00eR4rH'
    qm = 'https://www.youtube.com/playlist?list=PL8ER5-vAoiHAWm1UcZsiauUGPlJChgNXC'

    # Channels
    mleczny_czlowiek = r'https://www.youtube.com/@MleczneShoty/videos'
    pitala = 'https://www.youtube.com/@KacperPitala'
    astrofaza = 'https://www.youtube.com/@Astrofaza'
    everyday_hero = 'https://www.youtube.com/@EverydayHeroPL/videos'
    wszechnica = 'https://www.youtube.com/@WszechnicaFWW'
    poszukiwacz_okazji = 'https://www.youtube.com/@poszukiwaczeokazji'

    # Single clips
    wojenka = 'https://www.youtube.com/watch?v=O_OUUerrKfk&list=RDGMEMJQXQAmqrnmK1SEjY_rKBGAVMNpZHBMjyzN0&index=27'
    the_bill = 'https://www.youtube.com/watch?v=ZRrpVlSWlNI&list=RDGMEMJQXQAmqrnmK1SEjY_rKBGAVMNpZHBMjyzN0&index=32'
    monoteizm = 'https://www.youtube.com/watch?v=JKPmVvzr8uk'

    YtDownloader(
        urls=[
            US17,
            wojenka,
            the_bill,
            pitala,
        ]
    )
