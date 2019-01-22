from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer
import os
bot = ChatBot('My BOT');

trainer = ListTrainer(bot)
#bot.set_trainer(ListTrainer)

for files in os.listdir('/home/eddie/chabot_seq2seq/chatterbot-corpus/chatterbot_corpus/data/english/'):
    data = open('/home/eddie/chabot_seq2seq/chatterbot-corpus/chatterbot_corpus/data/english/' + files, 'r').readlines()
    trainer.train(data)


while True:
    message= input('You:')
    if message.strip() != 'Bye':
        reply = bot.get_response(message)
        print('ChatBot :', reply)
    if message.strip() == 'Bye':
        print('ChatBot : Bye')
        break



