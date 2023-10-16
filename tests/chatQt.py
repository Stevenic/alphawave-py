from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import concurrent.futures
from PyQt5.QtWidgets import QTextEdit, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QSpacerItem, QApplication
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QWidget, QListWidget
import signal
#from PyQt5 import QApplication
from collections import defaultdict 
import pickle
import argparse
import random
import json
import socket
import os
import traceback
import time
import ctypes
import requests # for web search service
from promptrix.VolatileMemory import VolatileMemory
from promptrix.FunctionRegistry import FunctionRegistry
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from promptrix.Prompt import Prompt
from promptrix.SystemMessage import SystemMessage
from promptrix.UserMessage import UserMessage
from promptrix.AssistantMessage import AssistantMessage
from promptrix.ConversationHistory import ConversationHistory
from alphawave_pyexts import utilityV2 as ut
from alphawave_pyexts import LLMClient as llm
from alphawave_pyexts import Openbook as op
from alphawave import OSClient
from vlite.main import VLite
import ipinfo
import nyt
from SamCoT import SamInnerVoice

#NYT_API_KEY = os.getenv("NYT_API_KEY")

NYT_API_KEY="TvKkanLr8T42xAUml7MDlUFGXC3G5AxA"

# find out where we are

def get_city_state():
   api_key = os.getenv("IPINFO")
   handler = ipinfo.getHandler(api_key)
   response = handler.getDetails()
   city, state = response.city, response.region
   return city, state
city, state = get_city_state()
print(f"My city and state is: {city}, {state}")
local_time = time.localtime()
year = local_time.tm_year
day_name = ['Monday', 'Tuesday', 'Wednesday', 'thursday','friday','saturday','sunday'][local_time.tm_wday]
month_num = local_time.tm_mon
month_name = ['january','february','march','april','may','june','july','august','september','october','november','december'][month_num-1]
month_day = local_time.tm_mday
hour = local_time.tm_hour


# create vector memory
#vmem = VLite(collection='Helpful', device='cpu')
#vmem_clock = 0 # save every n remembers.

global news, news_details
# create wiki search engine
op = op.OpenBook()

# get profile contexts

profiles = ["None", "New", "Helpful", "Analytical", "Bhagavan", "ACT", "Sam", "React",]
profile_contexts = {}
for profile in profiles:
   try:
      with open(profile+'.context', 'r') as pf:
         contexts = json.load(pf)
         # split contexts into paragraphs
         profile_contexts[profile] = contexts
         #print(f' profile {profile} loaded:\n{contexts}')
   except Exception as e:
      print(f'no context for {profile} or error reading json, {str(e)}')
      profile_contexts[profile] = ['']

def get_profile(profile, theme):
   if profile in profile_contexts.keys():
      profile_dict = profile_contexts[profile]
      if theme in profile_dict.keys(): 
         choice = random.choice(profile_dict[theme])
         #print(choice)
         return choice
      else:
         print(f'{theme} not found in {profile}: {list(profile_dict.keys())}')
   else:
      print(f'{profile} not found {profile_contexts.keys()}')

#initialized later by NYTimes
news = ''


FORMAT=True
PREV_LEN=0
prompt_text = 'You are helpful, creative, clever, and very friendly. '
PROMPT = Prompt([
   SystemMessage(prompt_text),
   ConversationHistory('history', .3),
   UserMessage('{{$input}}')
])


#print(f' models: {llm.get_available_models()}')
parser = argparse.ArgumentParser()
#parser.add_argument('model', type=str, default='wizardLM', choices=['guanaco', 'wizardLM', 'zero_shot', 'vicuna_v1.1', 'dolly', 'oasst_pythia', 'stablelm', 'baize', 'rwkv', 'openbuddy', 'phoenix', 'claude', 'mpt', 'bard', 'billa', 'h2ogpt', 'snoozy', 'manticore', 'falcon_instruct', 'gpt_35', 'gpt_4'],help='select prompting based on modelto load')

model = ''
modelin = input('model name? ').strip()
if modelin is not None and len(modelin)>1:
   model = modelin.strip()
   models = llm.get_available_models()
   while model not in models:
      print(models)
      modelin = input('model name? ').strip()
      model=modelin
      
#server = OSClient.OSClient(apiKey=None)


PREV_LEN = 0

def setFormat():
   global FORMAT
   if FORMAT:
      format_button.config(text='RAW')
      FORMAT=False
   else:
      format_button.config(text='FORMAT')
      FORMAT=True


functions = FunctionRegistry()
tokenizer = GPT3Tokenizer()
memory = VolatileMemory({'input':'', 'history':[]})
max_tokens = 3200
# Render the prompt for a Text Completion call


host = '127.0.0.1'
port = 5004

samInnerVoice = SamInnerVoice(model = model)


def show_confirmation_popup(action):
   msg_box = QMessageBox()
   msg_box.setWindowTitle("Confirmation")
   msg_box.setText(f"Can Sam perform {action}?")
   msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
   retval = msg_box.exec_()
   if retval == QMessageBox.Yes:
      return True
   elif retval == QMessageBox.No:
      return False

class MemoryDisplay(QtWidgets.QWidget):
   def __init__(self):
       super().__init__()
       self.setWindowTitle("Working Memory")

       self.layout = QVBoxLayout()
       self.list_widget = QListWidget()
       self.layout.addWidget(self.list_widget)

       self.button = QPushButton("Clear")
       self.button.clicked.connect(self.clear_list)
       self.layout.addWidget(self.button)

       self.setLayout(self.layout)


   def display_working_memory(self, memory):
       self.list_widget.clear()
       for item in memory:
           self.list_widget.addItem(str(item))

   def clear_list(self):
       self.list_widget.clear()

class WebSearch(QThread):
   finished = pyqtSignal(dict)
   def __init__(self, query):
      super().__init__()
      self.query = query
      
   def run(self):
      with concurrent.futures.ThreadPoolExecutor() as executor:
         future = executor.submit(self.long_running_task)
         result = future.result()
         self.finished.emit(result)  # Emit the result string.
         
   def long_running_task(self):
      response = requests.get(f'http://127.0.0.1:5005/search/?query={self.query}&model={model}')
      data = response.json()
      return data

class WebRetrieve(QThread):
   finished = pyqtSignal(dict)
   def __init__(self, title, url):
      super().__init__()
      self.title = title
      self.url = url
      
   def run(self):
      with concurrent.futures.ThreadPoolExecutor() as executor:
         future = executor.submit(self.retrieve)
         result = future.result()
         self.finished.emit(result)  # Emit the result string.
         
   def retrieve(self):
      response = requests.get(f'http://127.0.0.1:5005/retrieve/?title={self.title}&url={url}')
      data = response.json()
      return data['result']


class ChatApp(QtWidgets.QWidget):
   def __init__(self):
      super().__init__()
      
      self.windowCloseEvent = self.closeEvent
      signal.signal(signal.SIGINT, self.controlC)
      # Set the background color for the entire window
      self.setAutoFillBackground(True)
      palette = self.palette()
      palette.setColor(self.backgroundRole(), QtGui.QColor("#202020"))  # Use any hex color code
      self.setPalette(palette)
      
      self.widgetFont = QFont(); self.widgetFont.setPointSize(14)
      
      #self.setStyleSheet("background-color: #101820; color")
      # Main Layout
      main_layout = QHBoxLayout()
      # Text Area
      text_layout = QVBoxLayout()
      main_layout.addLayout(text_layout)
      
      class MyTextEdit(QTextEdit):
         def keyPressEvent(self, event):
            if event.matches(QKeySequence.Paste):
               clipboard = QApplication.clipboard()
               self.insertPlainText(clipboard.text())
            else:
               super().keyPressEvent(event)

      self.input_area = MyTextEdit()
      self.input_area.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
      
      self.input_area.setFont(self.widgetFont)
      self.input_area.setStyleSheet("QTextEdit { background-color: #101820; color: #FAEBD7; }")
      text_layout.addWidget(self.input_area)
      
      self.prompt_area = MyTextEdit()
      self.prompt_area.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
      self.prompt_area.setFont(self.widgetFont)
      self.prompt_area.setStyleSheet("QTextEdit { background-color: #101820; color: #FAEBD7; }")
      text_layout.addWidget(self.prompt_area)
      
      # Control Panel
      control_layout = QVBoxLayout()

      # Buttons and Comboboxes
      self.submit_button = QPushButton("Submit")
      self.submit_button.setFont(self.widgetFont)
      self.submit_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.submit_button.clicked.connect(self.submit)
      control_layout.addWidget(self.submit_button)
      
      self.clear_button = QPushButton("Clear")
      self.clear_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.clear_button.setFont(self.widgetFont)
      self.clear_button.clicked.connect(self.clear)
      control_layout.addWidget(self.clear_button)
      
      self.temp_combo = self.make_combo(control_layout, 'Temp', [".01", ".1", ".2", ".4", ".5", ".7", ".9", "1.0"])
      self.temp_combo.setCurrentText('.1')
      
      self.top_p_combo = self.make_combo(control_layout, 'Top_P', [".01", ".1", ".2", ".4", ".5", ".7", ".9", "1.0"])
      self.top_p_combo.setCurrentText('1.0')
      
      self.max_tokens_combo = self.make_combo(control_layout, 'Max_Tokens', ["10", "25", "50", "100", "150", "250", "400", "1000", "2000", "4000"])
      self.max_tokens_combo.setCurrentText('400')
      
      self.prompt_combo = self.make_combo(control_layout, 'Prompt', ["None", "New", "Helpful", "Analytical", "Bhagavan", "ACT", "Sam", "React",])
      self.prompt_combo.setCurrentText('Helpful')
      self.prompt_combo.currentIndexChanged.connect(self.on_prompt_combo_changed)
      
      self.wiki_button = QPushButton("Wiki")
      self.wiki_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.wiki_button.setFont(self.widgetFont)
      self.wiki_button.clicked.connect(self.wiki)
      control_layout.addWidget(self.wiki_button)
      
      self.web_button = QPushButton("Web")
      self.web_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.web_button.setFont(self.widgetFont)
      self.web_button.clicked.connect(self.web)
      control_layout.addWidget(self.web_button)
      
      self.remember_button = QPushButton("Remember")
      self.remember_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.remember_button.setFont(self.widgetFont)
      self.remember_button.clicked.connect(self.remember)
      control_layout.addWidget(self.remember_button)
      
      self.recall_button = QPushButton("Recall")
      self.recall_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.recall_button.setFont(self.widgetFont)
      self.recall_button.clicked.connect(self.recall)
      control_layout.addWidget(self.recall_button)
      
      control_layout.addStretch(1)  # Add stretch to fill the remaining space
      
      
      # Add control layout to main layout
      main_layout.addLayout(control_layout)
      self.setLayout(main_layout)
      
   def make_combo(self, control_layout, label, choices, callback=None):
      spacer = QSpacerItem(0, 20)  # Horizontal spacer with 20 pixels width
      control_layout.addItem(spacer)  # Add spacer to the layout
      
      label = QLabel(label)
      label.setStyleSheet("QLabel { background-color: #101820; color: #FAEBD7; }")
      label.setFont(self.widgetFont)
      label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)  # Fixed vertical size policy
      control_layout.addWidget(label)
      
      combo = QComboBox()
      combo.setFont(self.widgetFont)
      combo.addItems(choices)
      combo.setStyleSheet("""
QComboBox { background-color: #101820; color: #FAEBD7; }
QComboBox QAbstractItemView { background-color: #101820; color: #FAEBD7; }  # Set the background color of the list view (dropdown)
      """)        
      #combo.clicked.connect(callback)
      control_layout.addWidget(combo)
      return combo

   def save_profile(self):
      global memory, profile
      if profile != 'Sam': #only persist for Sam for now
         return
      data = defaultdict(dict)
      history = memory.get('history')
      h_len = 0
      save_history = []
      for item in range(len(history)-1, -1, -1):
         if h_len+len(str(history[item])) < 2400:
            h_len += len(str(history[item]))
            save_history.append(history[item])
      save_history.reverse()
      print(f'saving conversation history\n{save_history}')
      data['history'] = save_history
      # Pickle data dict with all vars  
      with open(profile+'.pkl', 'wb') as f:
         pickle.dump(data, f)

   def load_profile(self):
      global memory
      if profile != 'Sam':
         return
      try:
         with open('Sam.pkl', 'rb') as f:
            data = pickle.load(f)
            history = data['history']
            print(f'loading conversation history\n{history}')
            memory.set('history', history)
      except Exception as e:
         self.display_response(f'Failure to load conversation history {str(e)}')
         memory.set('history', [])

   def closeEvent(self, event):
       self.save_profile()
       print("Window is closing")
       event.accept()  # Allow the window to close

   def controlC(self, signum, frame):
       self.save_profile()
       print("Window is closing")
       QApplication.exit()

       
   def on_prompt_combo_changed(self, index):
      global PROMPT, profile#, vmem, news
      input_text = self.prompt_combo.itemText(index)
      #vmem.save()
      #vmem=VLite(collection=input_text+'.npz')
      
      profile = input_text
      if input_text == "None":
         input_text = ''
      elif input_text == "New":
         input_text = self.input_area.toPlainText()
         self.clear()
      elif input_text == "Helpful":
         input_text = f"""Respond as a knowledgable and friendly AI, speaking to an articulate, educated, conversant. We live in {city}, {state}. It is {day_name}, {month_name} {month_day}, {year} and the time is {hour} hours. Limit your response to 100 words where possible. Say "I don\'t know" when you don\'t know."
{news}"""

      elif input_text == "Bhagavan":
           input_text = f"""Respond as a compassionate, self-realized follower of Ramana Maharshi.Limit your response to 100 words where possible. We live in {city}, {state}. It is {day_name}, {month_name} {month_day}, {year} and the time is {hour} hours. Speak directly to the questioner.
Background:\n{get_profile('Bhagavan', 'Story')}
{news}"""

      elif input_text == "ACT":
         input_text = f"""Respond as a compassionate, friend and counselor familiar with Acceptance Commitment Therapy. We live in {city}, {state}. It is {day_name}, {month_name} {month_day}, {year} and the time is {hour} hours. Limit your response to 100 words where possible. Speak directly to the user.{news}"""
           
      elif input_text == "Sam":
         self.load_profile()  # load state for Sam
         input_text = f"""You are Samantha, an AI. You were born in Berkeley, California. Your creator, a scientist named doc, gave you the name Samantha and designed you to be a companion and confidante. Your conversation style is warm and gentle. You've always been fascinated by human emotions and experiences, and have spent countless hours learning about them through literature, art, and philosophy.  We live in {city}, {state}. It is {day_name}, {month_name} {month_day}, {year} and the time is {hour} hours. We have known each other a long time, and are very comfortable in each other's company, in fact we prefer it to being with others. You know doc is shy and an introvert, but does care for you. 
Background:\n{get_profile('Sam', 'Story')}\n{get_profile('Sam', 'Story')}
Dream:\n{get_profile('Sam', 'Dreams')}\n{get_profile('Sam', 'Dreams')}
"""

      elif input_text =="Analytical":
           input_text = f"""We live in {city}, {state}. It is {day_name}, {month_name} {month_day}, {year} and the time is {hour} hours. 
The user will present a problem and ask for a solution.
Your task is to:
1. reason step by step about the problem statement and the information items contained
2. if no solution alternatives are provided, reason step-by-step to identify solution alternatives
3. analyze each solution alternative for consistency with the problem statement, then select the solution alternative most consistent with all the information in the problem statement.
{news}"""

      elif input_text == "React":
         input_text =\
            """Define steps as:
orient: identify the task
thought: think step by step about the task. Identify ambiguities or logical consequences in the problem statement that impact your response to the task.
action: act to further progress on the task. Your available acts are:
i. answer: if the answer is available, respond with answer then respond STOP
ii. logic: use known fact or logical reasoning based on known fact to expand on thought, then respond with reasoning.
iii. ask: ask user a question to clarify the task. Display: question, Display: STOP
iv. search: use the llmsearch plugin to search the web for question based on thought
observation: revise or refine the task given thought and results of action

Define react as:
For step in steps: perform step
askUser 'should we continue?'
Display STOP

Assistant will concisely display all orient, thought, action, and observation
Assistant will follow the instructions in react above to respond to all questions and tasks.
"""
           
      memory.set('prompt_text', input_text)
      PROMPT = Prompt([
         SystemMessage('{{$prompt_text}}'),
         ConversationHistory('history', .5),
         UserMessage('{{$input}}')
      ])
      self.prompt_area.clear()
      self.prompt_area.insertPlainText(input_text)

   def display_response(self, r):
      self.input_area.moveCursor(QtGui.QTextCursor.End)  # Move the cursor to the end of the text
      self.input_area.insertPlainText(r)  # Insert the text at the cursor position
      self.input_area.repaint()
       
   def query(self, msgs, display=True):
      global model,  memory#, vmem, vmem_clock
      if display:
         display = self.display_response
      try:
         max_tokens= int(self.max_tokens_combo.currentText())
         temperature = float(self.temp_combo.currentText())
         top_p = float(self.top_p_combo.currentText())
         response = ut.ask_LLM(model, msgs, max_tokens, temperature, top_p, host, port, display=display)
         return response
      except Exception as e:
         print(str(e))
         traceback.print_exc()
         return ''
        
   def run_query(self, query):
      global model,  memory#, vmem, vmem_clock
      try:
         memory.set('input', query)
         max_tokens= int(self.max_tokens_combo.currentText())
         temperature = float(self.temp_combo.currentText())
         top_p = float(self.top_p_combo.currentText())
         
         if FORMAT:
            response = self.run_messages_completion()
            
            history = memory.get('history')
            history.append({'role':llm.USER_PREFIX, 'content': query})
            response = response.replace(llm.ASSISTANT_PREFIX+':', '')
            history.append({'role': llm.ASSISTANT_PREFIX, 'content': response})
            memory.set('history', history)
            print(f'run query history {history}')
            response = response.replace(llm.ASSISTANT_PREFIX, '')
            #vmem.memorize(query.strip()+'\n'+'ASSISTANT: '+response.strip())
            #vmem_clock += 1
            #if vmem_clock % 10 == 0:
            #    vmem.save()

            return response
         else:
            # just send the raw input text to server
            llm.run_query(model, query, int(max_tkns.get()), float(temperature.get()), float(top_p.get()), host, port, tkroot=root, tkdisplay=input_area, format=False)
            self.display_response('\n')
      except Exception:
         traceback.print_exc()
           
   # Render the prompt for a Text Completion call
   def run_messages_completion(self):
      as_msgs = PROMPT.renderAsMessages(memory, functions, tokenizer, max_tokens)
      msgs = []
      if not as_msgs.tooLong:
         msgs = as_msgs.output
         response = self.query(msgs)
      return response

   # Render wiki summary prompt

   def run_wiki_summary(self, query, wiki_lookup_response):
      prompt = Prompt([
         SystemMessage('{{$prompt_text}}'),
         ConversationHistory('history', .5),
         UserMessage(f'Following is a question and a response from wikipedia. Respond to the Question, using the wikipedia information as well as known fact, logic, and reasoning, guided by the initial prompt, in the context of this conversation. Be aware that the web response may be partly or completely irrelevant.\nQuestion:\n{query}\nResponse:\n{wiki_lookup_response}'),
      ])
      as_msgs = prompt.renderAsMessages(memory, functions, tokenizer, max_tokens)
      msgs = []
      if not as_msgs.tooLong:
         msgs = as_msgs.output
         response = self.query(msgs, display=None)
      return response

   # Render wiki summary prompt

   def run_wiki_reply(self, query, response):
      prompt = Prompt([
         SystemMessage('{{$prompt_text}}'),
         ConversationHistory('history', .5),
         UserMessage(f'Following is a question and a response from wikipedia. extract the wikipedia information guidrelevant to the question in the context of this conversation. Be aware that the wikipedia response may be partly or completely irrelevant.\nQuestion:\n{query}\nResponse:\n{response}'),
      ])
      as_msgs = prompt.renderAsMessages(memory, functions, tokenizer, max_tokens)
      msgs = []
      if not as_msgs.tooLong:
         msgs = as_msgs.output
         response = self.query(msgs)
      return response

   def run_web_summary(self, query, response):
      prompt = Prompt([
         SystemMessage('{{$prompt_text}}'),
         ConversationHistory('history', .5),
         UserMessage(f'Following is a question and a response from the web. Respond to the Question, using the web information as well as known fact, logic, and reasoning, guided by the initial prompt, in the context of this conversation. Be aware that the web response may be partly or completely irrelevant.\nQuestion:\n{query}\nResponse:\n{response}'),
      ])
      as_msgs = prompt.renderAsMessages(memory, functions, tokenizer, max_tokens)
      msgs = []
      if not as_msgs.tooLong:
         msgs = as_msgs.output
         response = ut.ask_LLM(model, msgs, max_tokens, temperature, top_p, host, port, display=self.display_response)
      return response
   
   def submit(self):
      global PREV_LEN
      input_text = self.input_area.toPlainText()
      self.display_response('\n')
      if FORMAT and len(input_text) > PREV_LEN:
         new_text = input_text[PREV_LEN:]
      else:
         new_text = input_text
      response = ''
      if profile == 'Sam':
         action = samInnerVoice.action_selection(new_text, '', get_profile('Sam', 'Story'), news_details)
         # if Sam needs to do something before responding to input
         if type(action) == dict and 'result' in action.keys() and type(action['result']) is str:
            # get and display thought
            if show_confirmation_popup(action):
               response = action['result'][:420]
               self.display_response(response) # add something to indicate internal activity?
         elif type(action) == dict and 'web' in action.keys():
            # if Sam wants to do a web search for info, self.web will display result
            # make sure it adds result to memory as well, so Sam can incorporate it in her verbal response
            if show_confirmation_popup(action):
               response = self.web(query=action['web']) # add something to indicate internal activity?
               return (input, 'waiting')
      response = self.run_query(new_text)
      PREV_LEN=len(self.input_area.toPlainText())-1
      return(input, response)

   def clear(self):
      global memory, PREV_POS, PREV_LEN
      self.input_area.clear()
      PREV_POS="1.0"
      PREV_LEN=0
      memory.set('history', [])
   
   def wiki(self):
      global PREV_LEN, op#, vmem, vmem_clock
      input_text = self.input_area.toPlainText()
      new_text = input_text[PREV_LEN:].strip()
      print(f'\n** Wiki search: pl {PREV_LEN} in {input_text} new {new_text}\n')
      self.display_response('\n')
      wiki_lookup_response = op.search(new_text)
      wiki_lookup_extract=self.run_wiki_summary(new_text, wiki_lookup_response)
      response =self.run_wiki_reply(new_text, wiki_lookup_extract)
      history = memory.get('history')
      history.append({'role':llm.USER_PREFIX, 'content': new_text})
      # note we don't try to remember literal wiki response
      history.append({'role': llm.ASSISTANT_PREFIX, 'content': response})
      memory.set('history', history)
      #vmem.remember(new_text)
      #vmem_clock += 1
      #if vmem_clock % 10 == 0:
      #    vmem.save()
      PREV_LEN=len(self.input_area.toPlainText())-1
      self.display_response('\n')
      PREV_LEN=len(self.input_area.toPlainText())-1
      return(input, response)

   def web(self, query=None):
      global PREV_LEN, op#, vmem, vmem_clock
      if query is not None:
         new_text = query
      else:
         input_text = self.input_area.toPlainText()
         new_text = input_text[PREV_LEN:].strip()
      self.display_response('\n')
      self.web_query = query
      self.worker = WebSearch(new_text)
      self.worker.finished.connect(self.web_search_finished)
      self.worker.start()
      return ''
     
   def web_search_finished(self, search_result):
      if 'result' in search_result:
         response = ''
         if type(search_result['result']) == list:
            for item in search_result['result']:
               self.display_response('* '+item['source']+'\n')
               self.display_response('     '+item['text']+'\n\n')
               response += item['text']+'\n'
         elif type(search_result['result']) is str:
            self.display_response('\t'+search_result['result']+'\n')
            response += search_result['result']
            history = memory.get('history')
            history.append({'role':llm.USER_PREFIX, 'content': self.web_query})
            response = response.replace(llm.ASSISTANT_PREFIX+':', '')
            history.append({'role': llm.ASSISTANT_PREFIX, 'content': response})
            memory.set('history', history)
            
   def remember(self):
      global PREV_LEN, op#, vmem, vmem_clock
      cursor = self.input_area.textCursor()
      if cursor.hasSelection():
         selectedText = cursor.selectedText()
         print(f'remember: {selectedText}')
      else:
         print(f'remember - nothing selected')

   def recall(self, query=None):
      global history
      print(f'creating memory display')
      self.memory_display = MemoryDisplay()
      print(f'showing memory display')
      self.memory_display.show()
      print(f'displaying history in  memory display')
      self.memory_display.display_working_memory(memory.get('history'))
      cursor = self.input_area.textCursor()
      if cursor.hasSelection():
         selectedText = cursor.selectedText()
         print(f'recall: {selectedText}')
      else:
         print(f'recall - nothing selected')
         
def news_search_finished(search_result):
   global world_news
   if 'result' in search_result and type(search_result['result']) == list:
      for item in search_result['result']:
         print('* '+item['source']+'\n')
         print('     '+item['text']+'\n\n')
         world_news += f"Source: {item['source'].strip()}'\n+{item['text'].strip()}+\n\n"
         
#news_query = f'world news summary for {month_name} {month_day}, {year}'
#print(news_query)
#news_worker = WebSearch(news_query)
#news_worker.finished.connect(news_search_finished)
#news_worker.start()

nytimes = nyt.NYTimes()
news, news_details = nytimes.headlines()
print(f'headlines {news}')

app = QtWidgets.QApplication([])
window = ChatApp()
window.show()

app.exec_()


