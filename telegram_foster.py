import telebot
import time

TOKEN = '5867649240:AAFFNvY7HOxZRJiBceO__mBp5cwH7KEFSJM'
bot = telebot.TeleBot(TOKEN)

def init_dict():
    tomato = 12
    cucamber = 15
    carrot = 10
    pepper = 4
    apple = 5
    orange = 4
    ginger = 1
    lemon = 4
    mint = 2
    basil = 2
    potato = 3
    sweet_potato = 3
    honey = 2
    sugar = 2

    foster_dict = {}
    foster_dict['מלפפון'] = cucamber
    foster_dict['מלפפונים'] = cucamber
    foster_dict['עגבנייה'] = tomato
    foster_dict['עגבניה'] = tomato
    foster_dict['עגבניות'] = tomato
    foster_dict['גזר'] = carrot
    foster_dict['גזרים'] = carrot
    foster_dict['גמבה'] = pepper
    foster_dict['גמבות'] = pepper
    foster_dict['פלפל'] = pepper
    foster_dict['פלפלים'] = pepper
    foster_dict['תפוח'] = apple
    foster_dict['תפוחים'] = apple
    foster_dict['תפוז'] = orange
    foster_dict['תפוזים'] = orange
    foster_dict['ג׳ינג׳ר'] = ginger
    foster_dict['גינגר'] = ginger
    foster_dict['ג׳ינגר'] = ginger
    foster_dict['גינג׳ר'] = ginger
    foster_dict['לימון'] = lemon
    foster_dict['לימונים'] = lemon
    foster_dict['נענע'] = mint
    foster_dict['בזיליקום'] = basil
    foster_dict['תפוח אדמה'] = potato
    foster_dict['תפו״א'] = potato
    foster_dict['תפוא'] = potato
    foster_dict['בטטה'] = sweet_potato
    return  foster_dict
foster_dict=init_dict()

@bot.message_handler(commands=['start', 'list', 'רשימה'])
def send_welcome(message):
	bot.reply_to(message, "Howdy, how are you doing?")

@bot.message_handler(func=lambda message: True)
def echo_all(message):
	if (message.text == 'רשימה'):
	    bot.add_callback_query_handler()

@bot.message_handler(commands=['veg'],content_types=['text'])
def handle_message(message):
    # Get the list of numbers from the message text
    #numbers = [int(x) for x in message.text.split(",")]
    bot.reply_to(message, "רשימת ירקנייה")
    foster_list = message.text.split("\n")
    main_str=''
    for item in foster_list:
        x=foster_list[0].split(' ')[0]
        y=int(foster_list[0].split(' ')[1])
        y_new=foster_dict[x]-y
        main_str+='{} {}\n'.format(y,x)
    # Send the processed numbers back to the user
    bot.send_message(message.chat.id, foster_list)

while True:
    try:
        bot.polling()
    except:
        time.sleep(15)