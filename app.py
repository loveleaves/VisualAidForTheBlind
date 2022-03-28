import os

os.system('git clone https://github.com/pytorch/fairseq.git; cd fairseq;'
          'pip install --use-feature=in-tree-build ./; cd ..')
os.system('ls -l')

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

# Register VQA task
tasks.register_task('vqa_gen',VqaGenTask)
# turn on cuda if GPU is available
use_cuda = torch.cuda.is_available()
# use fp16 only when GPU is available
use_fp16 = False

os.system('wget https://ofa-silicon.oss-us-west-1.aliyuncs.com/checkpoints/ofa_large_384.pt; '
          'mkdir -p checkpoints; mv ofa_large_384.pt checkpoints/ofa_large_384.pt')

# specify some options for evaluation
parser = options.get_generation_parser()
input_args = ["", "--task=vqa_gen", "--beam=100", "--unnormalized", "--path=checkpoints/ofa_large_384.pt", "--bpe-dir=utils/BPE"]
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


# Function for image captioning
def open_domain_vqa(Image, Question):
    sample = construct_sample(Image, Question)
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample
    # Run eval step for open-domain VQA
    with torch.no_grad():
        result, scores = zero_shot_step(task, generator, models, sample)
    return result[0]['answer']


title = "Visual Aid For The Blind"
description = "Gradio Demo for Visual Aid For The Blind. Upload your own image (high-resolution images are recommended) or click any one of the examples, and click " \
              "\"Submit\" and then wait for VFB's answer. "
article = "<p style='text-align: center'><a href='https://github.com/loveleaves/VisualAidForTheBlind' target='_blank'>VFB Github " \
          "Repo</a></p> "
examples = [['cat-4894153_1920.jpg', 'where are the cats?'], ['men-6245003_1920.jpg', 'how many people are in the image?'], ['labrador-retriever-7004193_1920.jpg', 'what breed is the dog in the picture?'], ['Starry_Night.jpeg', 'what style does the picture belong to?']]
io = gr.Interface(fn=open_domain_vqa, inputs=[gr.inputs.Image(type='pil'), "textbox"], outputs=gr.outputs.Textbox(label="Answer"),
                  title=title, description=description, article=article, examples=examples,
                  allow_flagging=False, allow_screenshot=False)
io.launch(cache_examples=True,share=True)
