from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO


def generate_mockup_img():
    mockup_img = Image.open('./pages/data/result_mockup.png')

    # 리사이즈할 크기 계산 (현재 크기의 반으로)
    width, height = mockup_img.size
    new_width = width // 2
    new_height = height // 2

    # 이미지 리사이즈
    mockup_img = mockup_img.resize((new_width, new_height))
    return mockup_img


def _generate_cover_img(img_url):
    # 이미지를 URL에서 가져오기
    img_url = img_url
    response = requests.get(img_url)

    # 이미지를 바이트 데이터로 읽기
    cover_img = BytesIO(response.content)

    # Pillow를 사용하여 이미지 열기
    cover_img = Image.open(cover_img)

    # Crop할 영역 설정 (left, upper, right, lower)
    crop_area = (7, 7, 263, 263)  # 예시: 왼쪽 위 (100, 100)에서 오른쪽 아래 (300, 300)까지 crop

    # 이미지 crop
    cover_img = cover_img.crop(crop_area)
    return cover_img


def _generate_title_img(title):
    # 책 제목 텍스트 및 폰트 설정
    text = title
    font_size = 19
    font_color = (255, 255, 255)
    font = ImageFont.truetype("./pages/data/HANBatangB.ttf", font_size)  # 폰트 선택 (폰트 파일 경로를 지정해야 함)

    # 책 제목 텍스트의 너비와 높이 계산
    text_bbox = font.getbbox(text)
    text_width = text_bbox[2]
    text_height = text_bbox[3]

    # 텍스트 길이가 길어질 경우 처리
    if text_width > 214:
        tmp = ''
        for string in text.split()[:2]:
            tmp += string + ' '
        text = tmp + '...'
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2]

    # 책 제목 배경 생성
    title_img = Image.new("RGBA", (text_width, text_height), (255, 255, 255, 0))

    # 배경 위에 텍스트 씌우기
    draw = ImageDraw.Draw(title_img)
    draw.text((0, 0), text, fill=font_color, font=font)
    return title_img


def _generate_authors_img(authors):
    # 작가 텍스트 및 폰트 설정
    text = authors
    font_size = 15
    font_color = (255, 255, 255)
    font = ImageFont.truetype("HANBatang.ttf", font_size)  # 폰트 선택 (폰트 파일 경로를 지정해야 함)

    # 작가 텍스트의 너비와 높이 계산
    text_bbox = font.getbbox(text)
    text_width = text_bbox[2]
    text_height = text_bbox[3]

    # 텍스트 길이가 길어질 경우 처리
    if text_width > 214:
        tmp = ''
        for string in text.split()[:2]:
            tmp += string + ' '
        text = tmp + '...'
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2]

    # 작가 배경 생성
    authors_img = Image.new("RGBA", (text_width, text_height), (255, 255, 255, 0))

    # 배경 위에 텍스트 씌우기
    draw = ImageDraw.Draw(authors_img)
    draw.text((0, 0), text, fill=font_color, font=font)
    return authors_img


def generate_result_img(index, mockup_img, img_url, title, authors):
    img1 = mockup_img.copy()
    img2 = _generate_cover_img(img_url)
    img3 = _generate_title_img(title)
    img4 = _generate_authors_img(authors)

    # 앨범 커버 위치 지정
    x_position = 21
    y_position = 23

    # mockup + cover
    img1.paste(img2, (x_position, y_position))

    # 타이틀 위치 지정
    x_position = 21
    y_position = 300

    # mockup + cover + title
    img1.paste(img3, (x_position, y_position), img3)

    # 작가 위치 지정
    x_position = 21  # x 좌표 조절
    y_position = 330   # y 좌표 조절

    # mockup + cover + title + authors
    img1.paste(img4, (x_position, y_position), img4)

    img1.save(f"./pages/result_img/result_{index}.png")
