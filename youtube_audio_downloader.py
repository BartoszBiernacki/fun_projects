from pytube import Channel, YouTube
from moviepy.editor import *
import os
from os.path import isfile, join
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


def remove_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
    else:
        print(f"Error while removing file. File {filename} does not exist")


def delete_all_files_in_given_directory_and_directory(directory, delete):
    if delete:
        for filename in os.listdir(directory):
            remove_file(filename=directory + "//" + filename)
        print("All files were removed from your local disk")
        os.rmdir(path=directory)
        print(f"Folder {directory} was removed from your local disk")


def get_first_itag(audio_streams):
    audio_streams_as_string = str(audio_streams)
    start = 16
    end = audio_streams_as_string.find('"', 16)
    itag = int(audio_streams_as_string[start:end])
    return itag


def download_audio_from_all_videos_on_channel(channel_url, output_path, N):
    print("\nDownloading mp4 (only audio) files from YouTube.")
    c = Channel(channel_url)
    i = 0
    for url in c.video_urls:
        yt = YouTube(url)
        itag = get_first_itag(audio_streams=yt.streams.filter(only_audio=True))
        stream = yt.streams.get_by_itag(itag)
        stream.download(output_path=output_path)
        i += 1
        print(f"Audio from video {i} out of {N} downloaded.")
    print(f"All {i} audios were downloaded.")


def convert_all_downloaded_mp4_to_mp3(output_path, N):
    print("\nConverting downloaded mp4 files to mp3.")
    i = 0
    for filename in os.listdir(output_path):
        if filename.endswith(".mp4"):
            filename = output_path + "//" + filename
            try:
                clip = AudioFileClip(filename)
                clip.write_audiofile(filename.replace(".mp4", ".mp3"), verbose=False, logger=None)
                clip.close()
                remove_file(filename)

                i += 1
                print(f"Converted {i} out of {N} files from mp4 to mp3")
            except:
                print(f"Problem with converting file {filename} to mp3")


def upload_files_to_my_google_drive(directory_that_contains_files, N):
    print("\nUploading mp3 files to Google Drive.")
    files = [f for f in os.listdir(directory_that_contains_files) if isfile(join(directory_that_contains_files, f))]
    
    gauth = GoogleAuth()
    drive = GoogleDrive(gauth)

    for i, file in enumerate(files):
        full_path_to_file = directory_that_contains_files + "//" + file
        gfile = drive.CreateFile({'parents': [{'id': '1G_M-O7mcci1ixkvcSWZ70l6VbyI7WL6I'}], 'title': file})
        # Read file and set it as the content of this instance.
        gfile.SetContentFile(full_path_to_file)
        gfile.Upload()  # Upload the file.
        print(f"File number: {i+1} out of {N} was uploaded")

    print(f"All {i+1} files were uploaded\n")
    print("Program finished.")


if __name__ == '__main__':
    yt_channel_url = "https://www.youtube.com/channel/UCBB9PNuQCHl4f2puROlotJQ/videos"
    yt_channel_name = "Natix"
    destination_folder = "C://Users//HAL//Downloads//YT_audio_downloads//" + yt_channel_name
    delete_files_after_uploading_on_google_drive = True

    number_of_movies_on_the_channel = len(Channel(yt_channel_url).video_urls)

    download_audio_from_all_videos_on_channel(channel_url=yt_channel_url, output_path=destination_folder, N=number_of_movies_on_the_channel)
    convert_all_downloaded_mp4_to_mp3(output_path=destination_folder, N=number_of_movies_on_the_channel)
    upload_files_to_my_google_drive(directory_that_contains_files=destination_folder, N=number_of_movies_on_the_channel)
    delete_all_files_in_given_directory_and_directory(directory=destination_folder, delete=delete_files_after_uploading_on_google_drive)
