#!/usr/bin/env python3
# coding=utf8
import requests
import nltk
import threading
from nltk.tokenize import sent_tokenize
import soundfile
import io
import logging
import socket
import re
import sys
import argparse
import os
from typing import List, Tuple
from flask import Flask
import librosa
from faster_whisper import WhisperModel
import numpy as np
from collections import deque

SAMPLING_RATE = 16000
USEPROMPT = False
PACKET_SIZE = 65536

# Server objects
loop = True
start = False


APIURL = "http://127.0.0.1:5555/translate"

# request to translate sentence


def translate(sentence: str, id: str):
    data = {'message': sentence, 'id': id}
    try:
        response = requests.post(args.apiurl, data=data, timeout=6)
    except requests.exceptions.Timeout:
        logging.info("Timeout")
        return
    answer = response.json()['message']
    # use id to find the original sentence location, and replace it with the translated sentence
    for i in range(len(sentences_maintain.translation_list)):
        if sentences_maintain.translation_list[i][1] == id:
            sentences_maintain.translation_list[i][0] = answer
            return
    # throw error
    logging.error("[ERROR] Translation id not found")
    logging.info("id: " + id)
    logging.info("sentence: " + sentence)
    logging.info("answer: " + answer)


def translate_tmp(sentence: str):
    data = {'message': sentence, 'id': '0'}
    try:
        response = requests.post(args.apiurl, data=data, timeout=6)
    except requests.exceptions.Timeout:
        logging.error("Timeout")
        return
    answer = response.json()['message']
    sentences_maintain.temp_translation = answer


# display functions
def display_all(sentence_list: List[Tuple[str, float]], temp):
    all_str = ' '.join(item[0] for item in sentence_list) + temp
    display_ansi(all_str)

try:
    from shutil import get_terminal_size
except:
    from backports.shutil_get_terminal_size import get_terminal_size

CN_REGEX = re.compile(u'[\u4e00-\u9fff]')
from functools import singledispatch
line_number = 0
@singledispatch
def display_ansi(content: List[str]):
    global line_number
    os.system('')
    columns = get_terminal_size().columns
    sys.stdout.write('\x1b[{}A\r'.format(line_number))
    line_number_tmp = sum(
        [((len(l) + len(CN_REGEX.findall(l))) // columns + 1) for l in content])

    # 打印空格来覆盖原来的内容
    print('\n'.join([' ' * columns] * line_number))
    sys.stdout.write('\x1b[{}A\r'.format(line_number))

    print('\n'.join(content))
    if line_number_tmp >= line_number:
        sys.stdout.write('\x1b[{}B'.format(line_number_tmp-line_number))
        line_number = line_number_tmp+1
    pass

@display_ansi.register
def _(content: str):
    display_ansi([content])


class Sentences_maintain:
    sentence_list = []
    translation_list = []
    temp_translation:str = ''
    pre_sentence:str = ''
    mid_sentence:str = ''
    pre_str:str = ''
    in_complete:str = ''
    lines = []
    all_you = True

    @staticmethod
    # 修改替换函数以在重复的短语后面添加 '...'
    def replace_repetitions(match: re.Match):
        # 获取匹配的字符串
        phrase = match.group(1)
        phrase = phrase.replace('...', '').strip()
        # 返回替换后的字符串
        return f"{phrase} ..."
    
    def replace_repetition(self,line:str):
        """compress repeating phrases"""
        pattern = re.compile(
            # r'(?<=\s)(\b[\w\.]+[\w\s\.]*[\w\.]\b)(?:\s?(?:\.\.\.)*\s\1)+?(?=\s|$)')
            r'(?<=\s)(\b[\w\.]+[\w\s\.,]*[\w\.]\b)(?:\s?(?:\.\.\.)*[\s\.,]\1)+?(?=\s|$)')
        while True:
            new_str, num_substitutions = pattern.subn(
                self.replace_repetitions, line)
            if num_substitutions == 0:
                break
            line = new_str
            while line.find('... ...') != -1 or line.find('......') != -1:
                line = line.replace(
                    '... ...', '...').replace('......', '...')
        line = pattern.sub(
            self.replace_repetitions, line)
        return line

    def input_line(self, line: str, time: float):
        line = line.replace('\r\n', '').replace('\n', '')
        
        line = line if line.startswith(
            self.mid_sentence) else self.mid_sentence + line
        # compress repeating phrases
        line = self.replace_repetition(line)
        
        sentences = sent_tokenize(line)
        if len(sentences) == 0:
            return
        # check if the last sentence is complete
        if sentences[-1].endswith(('.', '?', '!', ')', '♪')):
            self.pre_sentence = line
            self.mid_sentence = ''
        else:
            self.pre_sentence = ' '.join(
                sentences[:-1]) if len(sentences) > 1 else ''
            self.mid_sentence = sentences[-1]
        # compress repeating phrases
        self.pre_sentence = self.replace_repetition(self.pre_sentence)
        self.mid_sentence = self.replace_repetition(self.mid_sentence)
        # if sentece is not empty, add it to the list
        if len(self.pre_sentence.strip()) > 0 and (
                self.pre_sentence.strip() != '\n' or self.pre_sentence.strip() != '\r\n'):
            self.sentence_list.append(
                [self.pre_sentence.strip(), str(time)])
            self.translation_list.append(
                ["", str(time)])
            # translate the last complete sentence
            if len(self.translation_list) > 0:
                threading.Thread(
                    translate(self.sentence_list[-1][0], self.translation_list[-1][1])).start()
        
    
    def set_in_complete(self, in_complete: str):
        self.in_complete = in_complete
    
    def translate_tmp(self):
        if len(self.mid_sentence + self.in_complete) > 0:
            threading.Thread(
                translate_tmp(self.mid_sentence + self.in_complete)).start()


class WordsBuffer:
    def __init__(self):
        self.buffer = deque()
        self.new = deque()
        self.last_commited_time = 0

    def insert(self, new, offset):
        self.new = deque((start+offset, end+offset, text) for start, end, text in new if start+offset > self.last_commited_time-0.1)

    def flush(self):
        commit = []
        while self.new and self.buffer and self.new[0][2] == self.buffer[0][2]:
            commit.append(self.new[0])
            self.last_commited_time = self.new[0][1]
            self.buffer.popleft()
            self.new.popleft()
        self.buffer = self.new
        self.new = deque()
        return commit


class OnlineASRProcessor:

    def __init__(self, asr, buffer_trimming_sec=15):
        """asr: WhisperASR object\n
        buffer_trimming_sec: a number of seconds, buffer is trimmed if it is longer than "buffer_trimming_sec" threshold.\n
        """
        self.asr = asr
        self.buffer_trimming_sec = buffer_trimming_sec
        self.init()
        
    def init(self):
        """run this when starting or restarting processing"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset = 0
        self.transcript_buffer = WordsBuffer()
        self.committed = []

        self.silence_iters = 0

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    
    def prompt(self):
        """Returns a prompt, a 200-character suffix of commited text that is inside the scrolled away part of audio buffer. 
        """
        k = max(0, len(self.committed)-1)
        while k > 0 and self.committed[k-1][1] > self.buffer_time_offset:
            k -= 1

        p = [t for _, _, t in self.committed[:k]]
        p.reverse()

        l = 0
        prompt = [x for x in p if (l:=l+len(x)+1) <= 50]

        return "".join(prompt)

    def in_complete(self)->str:
        return str(self.to_flush(
            self.transcript_buffer.buffer)[2])
    
    def process_iter(self):
        """Runs on the current audio buffer.
        Returns: a tuple (beg_time, end_time, "text"), or (None, None, "").
        """
        if USEPROMPT:
            segments, _ = self.asr.transcribe(self.audio_buffer, language='en', initial_prompt=self.prompt(),
                                    beam_size=5, word_timestamps=True, condition_on_previous_text=True, **{})
        else:
            segments, _ = self.asr.transcribe(self.audio_buffer, language='en', beam_size=5,
                                    word_timestamps=True, condition_on_previous_text=True, **{})
        res = list(segments)

        # transform to [(beg,end,"word1"), ...]
        timeStampedWords = [(word.start, word.end, word.word) for segment in res for word in segment.words]

        self.transcript_buffer.insert(timeStampedWords, self.buffer_time_offset)
        out = self.transcript_buffer.flush()
        self.committed.extend(out)

        # there is a newly confirmed text
        if len(self.audio_buffer)/SAMPLING_RATE > self.buffer_trimming_sec:
            self.chunk_completed_segment(res)

        return self.to_flush(out)

    def chunk_completed_segment(self, res):
        """Trimming the audio buffer and updating the buffer time offset"""
        if not self.committed or len(res) <= 1:
            return
        
        t = self.committed[-1][1]
        ends = [s.end + self.buffer_time_offset for s in res]

        while len(ends) > 2 and ends[-2] > t:
            ends.pop(-1)
        e = ends[-2]

        if e <= t:
            delta_samples = int((e - self.buffer_time_offset) * SAMPLING_RATE)
            self.audio_buffer = self.audio_buffer[delta_samples:]
            self.buffer_time_offset = e

    @staticmethod
    def to_flush(sents, sep="", offset=0, ):
        """concatenate the timestamped words into one line
        sents: [(beg1, end1, "word1"), ...] or [] if empty
        return: (beg1,end-of-last-word,"concatenation of words") or (None, None, "")"""
        words = [w[2] for w in sents]
        if not words:
            return None, None, ""
        concatenated = sep.join(words)
        begin_timestamp = offset + sents[0][0]
        end_timestamp = offset + sents[-1][1]
        
        return begin_timestamp, end_timestamp, concatenated


class ServerProcessor:
    """wraps socket and ASR object, and serves one client connection"""
    def __init__(self, c, online_asr_proc:OnlineASRProcessor, min_chunk:float):
        self.connection = c
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk

    def receive_audio_chunk(self):
        """receive all audio that is available by this time\n
        receive more than self.min_chunk seconds"""
        out = []
        while sum(len(x) for x in out) < self.min_chunk*SAMPLING_RATE:
            raw_bytes = self.connection.recv(PACKET_SIZE)
            if not raw_bytes:
                break
            sf = soundfile.SoundFile(io.BytesIO(
                raw_bytes), channels=1, endian="LITTLE", samplerate=SAMPLING_RATE, subtype="PCM_16", format="RAW")
            audio, _ = librosa.load(sf, sr=SAMPLING_RATE)
            out.append(audio)
        if not out:
            return None
        return np.concatenate(out)

    def process(self):
        # handle one client connection
        self.online_asr_proc.init()
        while loop:
            audio = self.receive_audio_chunk()
            if audio is None:
                logging.info("break here")
                break
            self.online_asr_proc.insert_audio_chunk(audio)
            out = online.process_iter()
            # organize the output
            if out[0] is not None:
                sentences_maintain.input_line(out[2], out[0])
                sentences_maintain.set_in_complete(online.in_complete())
                sentences_maintain.translate_tmp()
                display_all(sentences_maintain.sentence_list,
                                    sentences_maintain.mid_sentence+sentences_maintain.in_complete)


def is_started():
    global start
    return start

def started():
    global start
    start = True

def stop_server():
    global loop
    loop = False

online: OnlineASRProcessor
sentences_maintain = Sentences_maintain()
def main_run():
    # 检查 punkt 数据包是否已经在本地存在
    nltk.download('punkt')
    # prepare, setting whisper object by args
    
    model_size_or_path = args.model_dir if args.model_dir else args.model
    if not model_size_or_path:
        raise ValueError("modelsize or model_dir parameter must be set")
    logging.info("Loading Whisper model from "+model_size_or_path)
    asr = WhisperModel(model_size_or_path, device="cuda",
                            compute_type="float16", download_root=args.model_cache_dir)

    global online
    online = OnlineASRProcessor(asr)

    if args.warmup_file and os.path.isfile(args.warmup_file):
        a = librosa.load(args.warmup_file, sr=SAMPLING_RATE)[0][0:1]
        asr.transcribe(a, language='en', beam_size=5, word_timestamps=True, condition_on_previous_text=True, **{})

    # server loop
    skt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    skt.bind((args.host1, args.port1))
    skt.listen(1)
    started()
    logging.info('INFO: Listening on'+str((args.host1, args.port1)))
    while loop:
        os.system('cls' if os.name == 'nt' else 'clear')
        conn, addr = skt.accept()
        logging.info('INFO: Connected to client on {}'.format(addr))
        conn.setblocking(True)
        proc = ServerProcessor(conn, online, args.min_chunk_size)
        proc.process()
        conn.close()
        logging.info('INFO: Connection to client closed')
    logging.info('INFO: Connection closed, terminating.')


app = Flask(__name__)

@app.route('/sentences', methods=['GET'])
def api():
    return ' '.join(item[0] for item in sentences_maintain.sentence_list)+' '+sentences_maintain.mid_sentence.strip()+' '+sentences_maintain.in_complete.strip()

@app.route('/translations', methods=['GET'])
def api2():
    return ' '.join(item[0] for item in sentences_maintain.translation_list)+sentences_maintain.temp_translation

@app.route('/')
def index():
    return "Hello, World!"
    

if __name__ == '__main__':
    # cli options
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-chunk-size', type=float, default=1.0,
                        help='Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.')
    parser.add_argument('--model', type=str, default='tiny', choices="tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large".split(
        ","), help="Name size of the Whisper model to use (default: tiny). The model is automatically downloaded from the model hub if not present in model cache dir.")
    parser.add_argument('--model_cache_dir', type=str, default=None,
                        help="Overriding the default model cache dir where models downloaded from the hub are saved")
    parser.add_argument('--model_dir', type=str, default=None,
                        help="Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.")
    parser.add_argument('--log', type=str, default='stderr',
                        help='Log file path. If not set, logs are sent to stderr.')
    parser.add_argument("--host1", type=str, default='localhost', help="The host of ASR server. Default is localhost.")
    parser.add_argument("--port1", type=int, default=43007, help="The port of ASR server. Default is 43007.")
    parser.add_argument("--warmup-file", type=str, default="test.wav", 
        help="The path to a speech audio wav file to warm up Whisper for faster processing.")
    parser.add_argument("--host2", type=str, default='localhost', help="The host of back-end server. Default is localhost.")
    parser.add_argument("--port2", type=int, default=6666, help="The port of back-end server. Default is 6666.")
    parser.add_argument("--apiurl",type=str, default=APIURL, help="The url of the translation api, default is "+APIURL)

    args = parser.parse_args()
    
    if args.log == 'stderr':
        args.log = sys.stderr
    else:
        args.log = open(args.log, 'a')
    
    # set up logging
    logging.basicConfig(level=logging.INFO, stream=args.log,
            format='%(asctime)s [line:%(lineno)d] - %(levelname)s: %(message)s')
    from flask.logging import default_handler
    app.logger.removeHandler(default_handler)
    handler = logging.StreamHandler(args.log)
    handler.setLevel(logging.WARN)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.WARN)
    logging.getLogger('werkzeug').setLevel(logging.WARN)
    
    # start two servers
    whisper = threading.Thread(target=main_run)
    whisper.start()
    app.run(port=args.port2, host=args.host2)
    stop_server()
    args.log.close()