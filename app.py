from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import os
import uuid
import json
from dotenv import load_dotenv
import google.genai as genai


# =========================
# CONFIG
# =========================
YOLO_CONF_THRESHOLD = 0.65
YOLO_MODEL_PATH = "./yolo11m.pt"

load_dotenv()

# =========================
# FASTAPI INIT
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD YOLO
# =========================
yolo_model = YOLO(YOLO_MODEL_PATH)

# =========================
# GEMINI CLIENT (NEW API)
# =========================
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# =========================
# OBJECT MAP
# =========================
object_map = {
    "person": {"chinese": "人", "pinyin": "rén"},
    "bicycle": {"chinese": "自行车", "pinyin": "zìxíngchē"},
    "car": {"chinese": "汽车", "pinyin": "qìchē"},
    "motorcycle": {"chinese": "摩托车", "pinyin": "mótuōchē"},
    "airplane": {"chinese": "飞机", "pinyin": "fēijī"},
    "bus": {"chinese": "公交车", "pinyin": "gōngjiāo chē"},
    "train": {"chinese": "火车", "pinyin": "huǒchē"},
    "truck": {"chinese": "卡车", "pinyin": "kǎchē"},
    "boat": {"chinese": "船", "pinyin": "chuán"},
    "traffic light": {"chinese": "红绿灯", "pinyin": "hónglǜdēng"},
    "fire hydrant": {"chinese": "消防栓", "pinyin": "xiāofáng shuān"},
    "stop sign": {"chinese": "停止标志", "pinyin": "tíngzhǐ biāozhì"},
    "parking meter": {"chinese": "停车计时器", "pinyin": "tíngchē jìshíqì"},
    "bench": {"chinese": "长椅", "pinyin": "chángyǐ"},
    "bird": {"chinese": "鸟", "pinyin": "niǎo"},
    "cat": {"chinese": "猫", "pinyin": "māo"},
    "dog": {"chinese": "狗", "pinyin": "gǒu"},
    "horse": {"chinese": "马", "pinyin": "mǎ"},
    "sheep": {"chinese": "羊", "pinyin": "yáng"},
    "cow": {"chinese": "牛", "pinyin": "niú"},
    "elephant": {"chinese": "大象", "pinyin": "dàxiàng"},
    "bear": {"chinese": "熊", "pinyin": "xióng"},
    "zebra": {"chinese": "斑马", "pinyin": "bānmǎ"},
    "giraffe": {"chinese": "长颈鹿", "pinyin": "chángjǐnglù"},
    "backpack": {"chinese": "背包", "pinyin": "bēibāo"},
    "umbrella": {"chinese": "雨伞", "pinyin": "yǔsǎn"},
    "handbag": {"chinese": "手提包", "pinyin": "shǒutíbāo"},
    "tie": {"chinese": "领带", "pinyin": "lǐngdài"},
    "suitcase": {"chinese": "行李箱", "pinyin": "xínglǐxiāng"},
    "frisbee": {"chinese": "飞盘", "pinyin": "fēipán"},
    "skis": {"chinese": "滑雪板", "pinyin": "huáxuěbǎn"},
    "snowboard": {"chinese": "单板滑雪", "pinyin": "dānbǎn huáxuě"},
    "sports ball": {"chinese": "运动球", "pinyin": "yùndòng qiú"},
    "kite": {"chinese": "风筝", "pinyin": "fēngzheng"},
    "baseball bat": {"chinese": "棒球棒", "pinyin": "bàngqiú bàng"},
    "baseball glove": {"chinese": "棒球手套", "pinyin": "bàngqiú shǒutào"},
    "skateboard": {"chinese": "滑板", "pinyin": "huábǎn"},
    "surfboard": {"chinese": "冲浪板", "pinyin": "chōnglàngbǎn"},
    "tennis racket": {"chinese": "网球拍", "pinyin": "wǎngqiú pāi"},
    "bottle": {"chinese": "瓶子", "pinyin": "píngzi"},
    "wine glass": {"chinese": "酒杯", "pinyin": "jiǔbēi"},
    "cup": {"chinese": "杯子", "pinyin": "bēizi"},
    "fork": {"chinese": "叉子", "pinyin": "chāzi"},
    "knife": {"chinese": "刀", "pinyin": "dāo"},
    "spoon": {"chinese": "勺子", "pinyin": "sháozi"},
    "bowl": {"chinese": "碗", "pinyin": "wǎn"},
    "banana": {"chinese": "香蕉", "pinyin": "xiāngjiāo"},
    "apple": {"chinese": "苹果", "pinyin": "píngguǒ"},
    "sandwich": {"chinese": "三明治", "pinyin": "sānmíngzhì"},
    "orange": {"chinese": "橙子", "pinyin": "chéngzi"},
    "broccoli": {"chinese": "西兰花", "pinyin": "xīlánhuā"},
    "carrot": {"chinese": "胡萝卜", "pinyin": "húluóbo"},
    "hot dog": {"chinese": "热狗", "pinyin": "règǒu"},
    "pizza": {"chinese": "披萨", "pinyin": "pīsà"},
    "donut": {"chinese": "甜甜圈", "pinyin": "tiántiánquān"},
    "cake": {"chinese": "蛋糕", "pinyin": "dàngāo"},
    "chair": {"chinese": "椅子", "pinyin": "yǐzi"},
    "couch": {"chinese": "沙发", "pinyin": "shāfā"},
    "potted plant": {"chinese": "盆栽", "pinyin": "pénzāi"},
    "bed": {"chinese": "床", "pinyin": "chuáng"},
    "dining table": {"chinese": "餐桌", "pinyin": "cānzhuō"},
    "toilet": {"chinese": "厕所", "pinyin": "cèsuǒ"},
    "tv": {"chinese": "电视", "pinyin": "diànshì"},
    "laptop": {"chinese": "笔记本电脑", "pinyin": "bǐjìběn diànnǎo"},
    "mouse": {"chinese": "鼠标", "pinyin": "shǔbiāo"},
    "remote": {"chinese": "遥控器", "pinyin": "yáokòngqì"},
    "keyboard": {"chinese": "键盘", "pinyin": "jiànpán"},
    "cell phone": {"chinese": "手机", "pinyin": "shǒujī"},
    "microwave": {"chinese": "微波炉", "pinyin": "wēibōlú"},
    "oven": {"chinese": "烤箱", "pinyin": "kǎoxiāng"},
    "toaster": {"chinese": "烤面包机", "pinyin": "kǎo miànbāo jī"},
    "sink": {"chinese": "水槽", "pinyin": "shuǐcáo"},
    "refrigerator": {"chinese": "冰箱", "pinyin": "bīngxiāng"},
    "book": {"chinese": "书", "pinyin": "shū"},
    "clock": {"chinese": "时钟", "pinyin": "shízhōng"},
    "vase": {"chinese": "花瓶", "pinyin": "huāpíng"},
    "scissors": {"chinese": "剪刀", "pinyin": "jiǎndāo"},
    "teddy bear": {"chinese": "泰迪熊", "pinyin": "tàidí xióng"},
    "hair drier": {"chinese": "吹风机", "pinyin": "chuīfēngjī"},
    "toothbrush": {"chinese": "牙刷", "pinyin": "yáshuā"}
}

LIVING_CLASSES = {
    "person",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe"
}
# =========================
# GEMINI TEXT (YOLO SUCCESS)
# =========================
def generate_usage_sentence(label: str) -> str:
    label_lower = label.lower()

    if label_lower in LIVING_CLASSES:
        prompt = (
            f"Explain what a {label} is in very simple English. "
            f"Talk about what it does or how it lives. "
            f"Use easy daily words like explaining to a child. "
            f"response should be of max 2 lines. "
            f"Do NOT say 'used for'. "
            f"Do not mention colors, size, or images."
        )
    else:
        prompt = (
            f"Explain what a {label} is used for using very simple English. "
            f"List common uses separated by commas. "
            f"Use easy daily words. "
            f"response should be of max 2 lines. "
            f"Do not use technical words. "
            f"Do not mention color, size, or images."
        )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text.strip()


# =========================
# GEMINI VISION (YOLO FAIL)
# =========================
def analyze_with_gemini(image_path: str):
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_bytes
                        }
                    },
                    {
"text": (
    "Identify the main object in this image and return ONLY valid JSON:\n"
    "{"
    "\"object\":\"English name\","
    "\"chinese\":\"Chinese (simplified)\","
    "\"pinyin\":\"Pinyin with tones\","
    "\"sentence\":\"Very simple sentence using easy daily words. "
    "If the object is a thing, explain what people use it for in real life. "
    "If the object is a person or animal, explain what it does or how it lives. "
    "Use short phrases separated by commas. "
    "No technical or formal words.\""
    "}\n"
    "Rules:\n"
    "- Use very easy English (like explaining to a child)\n"
    "- For people or animals, do NOT say 'used for'\n"
    "- Do NOT use scientific or technical words\n"
    "- Do NOT mention color, size, or the image\n"
    "- Keep it practical and simple\n"
    "- response should be of max 2 lines."
)


                    }
                ]
            }
        ]
    )

    text = response.text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except:
        return {
            "object": "Unknown",
            "chinese": "未知",
            "pinyin": "wèizhī",
            "sentence": response.text.strip()
        }

# =========================
# API ENDPOINT
# =========================
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    temp_path = f"temp_{uuid.uuid4().hex}.jpg"

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    frame = cv2.imread(temp_path)
    results = yolo_model(frame)

    best_conf = 0.0
    best_label = None

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_label = yolo_model.names[int(box.cls[0])]

    # =========================
    # DECISION LOGIC
    # =========================
    if best_label and best_conf >= YOLO_CONF_THRESHOLD:
        chinese_info = object_map.get(
            best_label.lower(),
            {"chinese": "未知", "pinyin": "---"}
        )

        sentence = generate_usage_sentence(best_label)

        result = {
            "object": best_label,
            "chinese": chinese_info["chinese"],
            "pinyin": chinese_info["pinyin"],
            "sentence": sentence,
            "source": "YOLO + Gemini (Text)"
        }
    else:
        gemini_result = analyze_with_gemini(temp_path)

        label = gemini_result.get("object", "Unknown").lower()
        chinese_info = object_map.get(
            label,
            {
                "chinese": gemini_result.get("chinese", "未知"),
                "pinyin": gemini_result.get("pinyin", "---")
            }
        )

        result = {
            "object": label,
            "chinese": chinese_info["chinese"],
            "pinyin": chinese_info["pinyin"],
            "sentence": gemini_result.get("sentence"),
            "source": "Gemini Vision Fallback"
        }

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return result

