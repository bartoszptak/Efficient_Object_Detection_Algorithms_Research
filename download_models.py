import requests
import zipfile
import os


def download_file_from_google_drive(id:str, destination:str) -> None:
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    print('GET: {}'.format(destination))
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def download_file_from_link(link:str, destination:str) -> None:
    r = requests.get(link, allow_redirects=True)
    open(destination, 'wb').write(r.content)

def main() -> None:
    print('[LOGS] Download YoloV3-4classes')
    # download YoloV3 from darknet
    download_file_from_google_drive(
        '1eWkLwTY_WnC5g55gjwdhy4HARKwNsUgT',
        'models/YOLOv3/YOLOv3.weights'
    )

    print('[LOGS] Download EfficientDet-d0-4classes')
    # download from darknet
    download_file_from_google_drive(
        '1UWr5Nbamtnnu7i-RYWEC8S10GhhGpxEl',
        'models/EfficientDet/EfficientDet-d0/EfficientDet-d0.weights'
    )

if __name__ == "__main__":
    main()
    
