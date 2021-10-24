from pytube import Channel, YouTube
from moviepy.editor import *
import os
from os.path import isfile, join
import time


def remove_file(filename, directory=None):
    if directory is None:
        full_path = filename
    else:
        full_path = directory + "//" + filename
    try:
        os.remove(full_path)
    except Exception:
        print(f"Error while removing file. File {full_path} does not exist")


def remove_file_with_given_prefix(directory, prefix):
    filenames = [f for f in os.listdir(directory) if isfile(join(directory, f))]
    for filename in filenames:
        if filename.startswith(prefix):
            remove_file(directory=directory, filename=filename)
            break


def get_number_of_movies_on_channel(YouTube_channel_url):
    while True:
        try:
            number_of_movies_on_the_channel = len(Channel(YouTube_channel_url).video_urls)
            return number_of_movies_on_the_channel
        except Exception:
            pass


def get_first_itag(audio_streams):
    audio_streams_as_string = str(audio_streams)
    start = 16
    end = audio_streams_as_string.find('"', 16)
    itag = int(audio_streams_as_string[start:end])
    return itag


def convert_mp4_file_to_mp3(filename):
    if filename.endswith(".mp4"):
        try:
            clip = AudioFileClip(filename)
            clip.write_audiofile(filename.replace(".mp4", ".mp3"), verbose=False, logger=None)
            clip.close()
            remove_file(filename=filename)
        except Exception:
            print(f"Problem with converting file {filename} to mp3")


def download_audio_from_yt_video_as_mp4(url, output_path, file_prefix):
    while True:
        try:
            yt = YouTube(url)
            itag = get_first_itag(audio_streams=yt.streams.filter(only_audio=True))
            stream = yt.streams.get_by_itag(itag)
            filename = stream.download(output_path=output_path, filename_prefix=file_prefix)
            passed = True
        except Exception:
            time.sleep(1)
            remove_file_with_given_prefix(directory=output_path, prefix=file_prefix)
            passed = False
        if passed:
            break
    return filename


def download_audio_from_all_videos_on_channel(channel_url, output_path, N, file_prefix):
    print("\nDownloading mp4 (only audio) files from YouTube.")
    c = Channel(channel_url)
    i = 0
    for url in reversed(list(c.video_urls)):
        dynamic_prefix = file_prefix + "[" + str(f"{i+1:0{len(str(N))}d}") + "] "
        print(f"Downloading audio from video {i+1} out of {N} ...")
        filename = download_audio_from_yt_video_as_mp4(url=url, output_path=output_path, file_prefix=dynamic_prefix)
        print(f"Converting from mp4 to mp3 file {i+1} out of {N} ...")
        convert_mp4_file_to_mp3(filename=filename)
        print()
        i += 1
    print(f"All {i} audios were downloaded.")


if __name__ == '__main__':
    yt_channel_name = "Test"
    yt_channel_url = "https://www.youtube.com/c/Sonikkua/videos"
    destination_folder = "C://Users//HAL//Downloads//YT_audio_downloads//" + yt_channel_name
    # modify only lines above ---------------------------------------------------------------------

    print("Main program started.")
    num_of_movies = get_number_of_movies_on_channel(YouTube_channel_url=yt_channel_url)
    print(f"There are {num_of_movies} movies on that channel")

    download_audio_from_all_videos_on_channel(channel_url=yt_channel_url, output_path=destination_folder,
                                              N=num_of_movies, file_prefix=yt_channel_name + " ")
