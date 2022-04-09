import os

os.system('git clone https://github.com/pytorch/fairseq.git; cd fairseq;'
          'pip install --use-feature=in-tree-build ./; cd ..')
os.system('curl -L ip.tool.lu; pip install torchvision; pip install opencv-python-headless')
os.system('bash vqa_large_best.sh; mkdir -p checkpoints; mv vqa_large_best.pt checkpoints/vqa.pt')

import torch
import numpy as np
import re
from fairseq import utils,tasks
from fairseq import checkpoint_utils
from fairseq import distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from utils.zero_shot_utils import zero_shot_step
from tasks.mm_tasks.vqa_gen import VqaGenTask
from models.ofa import OFAModel
from PIL import Image
from torchvision import transforms
import gradio as gr
import sys
import uuid
import requests
import wave
import base64
import hashlib
from imp import reload
import time

reload(sys)

APP_KEY = '0f3e5006d4c9e72a'
APP_SECRET = 'H7zqPQyJlTOPxVmfvFVeMNcolKxQXREF'

# Register VQA task
tasks.register_task('vqa_gen',VqaGenTask)
# turn on cuda if GPU is available
use_cuda = torch.cuda.is_available()
# use fp16 only when GPU is available
use_fp16 = False

# specify some options for evaluation
parser = options.get_generation_parser()
input_args = ["", "--task=vqa_gen", "--beam=100", "--unnormalized", "--path=checkpoints/vqa.pt", "--bpe-dir=utils/BPE"]
args = options.parse_args_and_arch(parser, input_args)
cfg = convert_namespace_to_omegaconf(args)

# Load pretrained ckpt & config
task = tasks.setup_task(cfg.task)
models, cfg = checkpoint_utils.load_model_ensemble(
    utils.split_paths(cfg.common_eval.path),
    task=task
)

# Move models to GPU
for model in models:
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

# Initialize generator
generator = task.build_generator(models, cfg.generation)

# Image transform
from torchvision import transforms
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Text preprocess
bos_item = torch.LongTensor([task.src_dict.bos()])
eos_item = torch.LongTensor([task.src_dict.eos()])
pad_idx = task.src_dict.pad()

# Normalize the question
def pre_question(question, max_ques_words):
    question = question.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ')
    question = re.sub(
        r"\s{2,}",
        ' ',
        question,
    )
    question = question.rstrip('\n')
    question = question.strip(' ')
    # truncate question
    question_words = question.split(' ')
    if len(question_words) > max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
    return question

def encode_text(text, length=None, append_bos=False, append_eos=False):
    s = task.tgt_dict.encode_line(
        line=task.bpe.encode(text),
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s

# Construct input for open-domain VQA task
def construct_sample(image: Image, question: str):
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])

    question = pre_question(question, task.cfg.max_src_length)
    question = question + '?' if not question.endswith('?') else question
    src_text = encode_text(' {}'.format(question), append_bos=True, append_eos=True).unsqueeze(0)

    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    ref_dict = np.array([{'yes': 1.0}]) # just placeholder
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask,
        },
        "ref_dict": ref_dict,
    }
    return sample
  
# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t

# Audio recognize
def truncate(q):
    if q is None:
        return None
    size = len(q)
    return q if size <= 20 else q[0:10] + str(size) + q[size-10:size]

def encrypt(signStr):
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(signStr.encode('utf-8'))
    return hash_algorithm.hexdigest()

def audio_recognize(audio_file_path):
    lang_type = 'zh-CHS'
    extension = audio_file_path[audio_file_path.rindex('.')+1:]
    if extension != 'wav':
        print('不支持的音频类型')
        sys.exit(1)
    wav_info = wave.open(audio_file_path, 'rb')
    sample_rate = wav_info.getframerate()
    nchannels = wav_info.getnchannels()
    wav_info.close()
    with open(audio_file_path, 'rb') as file_wav:
        q = base64.b64encode(file_wav.read()).decode('utf-8')

    data = {}
    curtime = str(int(time.time()))
    data['curtime'] = curtime
    salt = str(uuid.uuid1())
    signStr = APP_KEY + truncate(q) + salt + curtime + APP_SECRET
    sign = encrypt(signStr)
    data['appKey'] = APP_KEY
    data['q'] = q
    data['salt'] = salt
    data['sign'] = sign
    data['signType'] = "v2"
    data['langType'] = lang_type
    data['rate'] = sample_rate
    data['format'] = 'wav'
    data['channel'] = nchannels
    data['type'] = 1

    # 数据请求
    YOUDAO_AUDIO_RECOGNIZE_URL = 'https://openapi.youdao.com/asrapi'
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(YOUDAO_AUDIO_RECOGNIZE_URL, data=data, headers=headers)
    answer = response.content
    true = 1
    false = 0
    str1=str(answer, encoding = "utf-8")
    # print(str1)
    answer=eval(str1)
    answer = answer["result"][0]
    # print(answer)
    return answer

def resample_rate(path,new_sample_rate = 16000): # 用于改变音频采样频率

    signal, sr = librosa.load(path, sr=None)
    wavfile = path.split('/')[-1]
    wavfile = wavfile.split('.')[0]
    file_name = wavfile + '_new.wav'
    new_signal = librosa.resample(signal, sr, new_sample_rate) # 
    librosa.output.write_wav(file_name, new_signal , new_sample_rate)

def handle_audio(audio):
    sr, data = audio
    file_path = 'temp_audio.wav'
    new_file_path = "temp_audio_new.wav"
    if os.path.isfile(file_path):
      os.system("rm -rf " + file_path)
    wavfile = wave.open(file_path, 'wb')
    # 以下是wav音频参数
    wavfile.setnchannels(1)
    wavfile.setsampwidth(32 // 8)
    wavfile.setframerate(sr)
    wavfile.writeframes(data)
    wavfile.close()
    # resample_rate('a.wav')
    # os.system("ffmpeg -i 'temp_audio.wav' -ar 16000 'temp_audio_new.wav'")
    song = AudioSegment.from_wav(file_path).set_frame_rate(16000)
    song.export(new_file_path, format="wav")
    # os.system('rm -rf temp_audio.wav') # 删除缓存音频文件
    audio_answer = audio_recognize(new_file_path)
    os.system('rm -rf temp_audio_new.wav') # 删除缓存音频文件
    return audio_answer

# sentence translation
def sentence_trans(q,Trans_to = "auto"):
    data = {}
    data['from'] = 'auto'
    data['to'] = Trans_to
    data['signType'] = 'v3'
    curtime = str(int(time.time()))
    data['curtime'] = curtime
    salt = str(uuid.uuid1())
    signStr = APP_KEY + truncate(q) + salt + curtime + APP_SECRET
    sign = encrypt(signStr)
    data['appKey'] = APP_KEY
    data['q'] = q
    data['salt'] = salt
    data['sign'] = sign
    data['vocabId'] = "2D4552567D81424D91FBF4805C70E05A"

    # 数据请求
    YOUDAO_SENTENCE_URL = 'https://openapi.youdao.com/api'
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(YOUDAO_SENTENCE_URL, data=data, headers=headers)
    contentType = response.headers['Content-Type']
    content = str(response.content,'utf-8')
    # print(content)
    false = 0
    true = 1 # 用于处理str转dict时key值为true值未定义的情况
    content = eval(content)
    answer = ""
    try:
        answer = content["translation"][0]
    except Exception as e:
        answer = content["web"][0]["value"][0]
    # print(answer)
    return answer

# audio generate
def audio_generate_encrypt(signStr):
    hash_algorithm = hashlib.md5()
    hash_algorithm.update(signStr.encode('utf-8'))
    return hash_algorithm.hexdigest()

def audio_generate(q):
    data = {}
    data['langType'] = 'zh-CHS'
    salt = str(uuid.uuid1())
    signStr = APP_KEY + q + salt + APP_SECRET
    sign = audio_generate_encrypt(signStr)
    data['appKey'] = APP_KEY
    data['q'] = q
    data['salt'] = salt
    data['sign'] = sign

    # 数据请求
    YOUDAO_AUDIO_GENERATE_URL = 'https://openapi.youdao.com/ttsapi'
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(YOUDAO_AUDIO_GENERATE_URL, data=data, headers=headers)
    contentType = response.headers['Content-Type']
    # millis = int(round(time.time() * 1000))
    # filePath = "audio" + str(millis) + ".mp3"
    filePath = "audio_answer.mp3"
    if os.path.isfile(filePath):
        os.system("rm -rf " + filePath)
    fo = open(filePath, 'wb')
    fo.write(response.content)
    fo.close()
    # audio_answer = AudioSegment.from_file("audio_answer.mp3", format = 'MP3')
    # os.system("ffmpeg -i 'temp_audio.wav' -ar 16000 'temp_audio_new.wav'")
    # audio_answer = list(audio_answer._data)
    # audio_answer = np.array(audio_answer)
    # print(audio_answer)
    # return (48000,audio_answer)
    return filePath

# Function for image captioning
def open_domain_vqa(Image, Question):
    # preprocess data
    input_question = sentence_trans(Question,"en")
    # put data into model
    sample = construct_sample(Image, input_question)
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample
    # Run eval step for open-domain VQA
    with torch.no_grad():
        result, scores = zero_shot_step(task, generator, models, sample)
    answer = result[0]['answer']
    # preprocess answer
    sentence_answer = sentence_trans(answer,"zh-CHS")
    audio_answer = audio_generate(sentence_answer)
    # return audio answer
    # sr,data = audio_answer
    # return (sr,data)
    return (sentence_answer, audio_answer)


title = "盲人避障系统"
description = "盲人避障系统的Demo。 食用指南：上传一张图片 (建议使用高分辨率图像) 和一句话的问题, 然后点击 " \
              "\"Submit\" ，等待些许即可得到 VFB's 回答结果。"
article = "<p style='text-align: center'><a href='https://github.com/loveleaves/VisualAidForTheBlind' target='_blank'>VFB Github " \
          "Repo</a></p> "
examples = [['example-1.jpg', '图片里有多少只猫?'], ['example-2.jpg', '图片里有多少个人?'], ['example-3.jpg', '图片中的狗是什么品种?'], ['example-4.jpeg', '这张图片属于什么风格?']]
io = gr.Interface(fn=open_domain_vqa, inputs=[gr.inputs.Image(type='pil'), "textbox"], outputs=[gr.outputs.Textbox(label="Caption"),gr.outputs.Audio(type="file")],
                  title=title, description=description, article=article, examples=examples,
                  allow_flagging=False, allow_screenshot=False)
io.launch(cache_examples=True)