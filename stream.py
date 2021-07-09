import streamlit as st
import numpy as np
import os
import urllib
import glob
import SessionState
import cv2

# DATA_URL_ROOT = st.secrets['DATA_URL_ROOT']
# DATA_URL_ROOT = "data/"
DATA_URL_ROOT = (
    "https://raw.githubusercontent.com/sweetcocoa/streamlit_image_explorer/master/"
)

session_state = SessionState.get(image_idx=0)
files = dict(train=list(), val=list())
data_split = "train"
test_on_local = False



if test_on_local:

    def get_file_list(base_path):
        images = sorted(glob.glob(f"{base_path}/**/*.png", recursive=True))
        images = [image.replace("\\", "/") for image in images]
        return images

    @st.cache(show_spinner=False)
    def get_file_content_as_string(path):
        return open(path, "r").read()

    @st.cache(show_spinner=False)
    def load_image(url, resize=None):
        image = cv2.imread(url, cv2.IMREAD_COLOR)
        if resize is not None:
            image = cv2.resize(image, dsize=resize, interpolation=cv2.INTER_LINEAR)

        image = image[:, :, [2, 1, 0]]  # BGR -> RGB
        return image


else:

    def get_file_list(base_path):
        images = sorted(glob.glob(f"{base_path}/**/*.png", recursive=True))
        images = [(DATA_URL_ROOT + image).replace("\\", "/") for image in images]
        return images

    @st.cache(show_spinner=False)
    def get_file_content_as_string(path):
        global DATA_URL_ROOT
        url = DATA_URL_ROOT + path
        response = urllib.request.urlopen(url)
        return response.read().decode("utf-8")

    @st.cache(show_spinner=False)
    def load_image(url, resize=None):
        with urllib.request.urlopen(url) as response:
            image = np.asarray(bytearray(response.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if resize is not None:
            image = cv2.resize(image, dsize=resize, interpolation=cv2.INTER_LINEAR)

        image = image[:, :, [2, 1, 0]]  # BGR -> RGB
        return image


def label_of(path):
    return path.split("/")[-2]


def split_of(path):
    # print(path)
    return path.split("/")[-3]


def image_explorer():
    global session_state, data_split, files

    image_idx = session_state.image_idx

    title_columns = st.beta_columns(2)
    data_split = title_columns[0].radio("Choose data split", ("train", "val"))
    is_resized = title_columns[1].checkbox(
        "Resize",
        value=False,
    )

    # data_split = st.
    num_images_row = st.slider(
        "Number of Images in a Row",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
        format=None,
        key=None,
        help=None,
    )
    num_images_col = 5
    number_of_images_in_page = int(num_images_col * num_images_row)

    exploer_buttons = st.beta_columns(2)
    prev_button = exploer_buttons[0].button("Prev Images")
    next_button = exploer_buttons[1].button("Next Images")

    if prev_button:
        image_idx = max(image_idx - number_of_images_in_page, 0)
        session_state.image_idx = image_idx
    if next_button:
        image_idx = min(image_idx + number_of_images_in_page, len(files[data_split]))
        session_state.image_idx = image_idx

    st.header(
        f"Images from {image_idx} to {min(len(files[data_split]), image_idx + num_images_col * (num_images_row))} / {len(files[data_split])}"
    )
    columns = st.beta_columns(num_images_row)
    for i in range(len(columns)):
        start_idx = image_idx + i * num_images_col
        end_idx = min(start_idx + num_images_col, len(files[data_split]))

        # print(start_idx, end_idx)
        if not is_resized:
            columns[i].image(
                files[data_split][start_idx:end_idx],
                caption=[
                    f"{label_of(files[data_split][i])}, {i}"
                    for i in range(start_idx, end_idx)
                ],
            )
        else:
            columns[i].image(
                [
                    load_image(files[data_split][i], resize=(32, 32))
                    for i in range(start_idx, end_idx)
                ],
                caption=[
                    f"{label_of(files[data_split][i])}, {i}"
                    for i in range(start_idx, end_idx)
                ],
            )


def main():
    global files
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("Image Explorer")
    front_text = st.markdown(get_file_content_as_string("front.md"))

    _files = get_file_list("data/")

    for file in _files:
        split = split_of(file)
        files[split].append(file)

    app_mode = st.sidebar.selectbox(
        "Choose the app mode", ["Show instructions", "Launch", "Show the source code"]
    )
    if app_mode == "Show instructions":
        st.sidebar.success('To Launch Explorer, Select "Launch".')

    elif app_mode == "Show the source code":
        front_text.empty()
        st.code(get_file_content_as_string("stream.py"))
    elif app_mode == "Launch":
        front_text.empty()
        image_explorer()


if __name__ == "__main__":
    main()
