#import bottle to handle connections with HourAI
from bottle import get, post, request, run
#for loading the AI model/tokenizer
from transformers import AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer
import torch #for the chat history

#globals
args = 'mlem'
tokenizer = 'blep'
model = 'uwu'
chat_hist = [{'test': 'owo'}]

#define check function - checks is ShanghAI is able to communicate with HourAI
##decorator binds a piece of code to an URL path
##in this case, we link the /check path to the check() function
@post('/check', method='POST')
def check():
    print('HourAI connected!')
    reply = 'check'
    return reply

#define load function - gets model and loads in into memory
##decorator binds a piece of code to an URL path
##in this case, we link the /load path to the load() function
@post('/load', method='POST')
def load():
    #get model name from load request
    model_name = request.json.get('model')
    global args, tokenizer, model
    args = request.json.get('args')
    #start by loading up the model/tokenizer
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print('Tokenizer loaded.')
    print('Loading model...')
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print('Model loaded.')
    #choose device to run on (GPU or CPU)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = model.to(device)
    #print('Model loaded on {}'.format(device))
    ####only for testing on non CUDA devices
    ####force model to run on CPU
    model.cpu()
    #create the ready reply
    reply = "ShanghAI: Model {} ready!".format(model_name)
    #send the reply over to HourAI
    return reply

#define generate function - generates a response to a given text input
##decorator binds a piece of code to an URL path
##in this case, we link the /generate path to the generate() function
@post('/generate', method='POST')
def generate():
    #get model name from generate request
    input = request.json.get('inputs')
    #get channel for chat history
    channel_id = request.json.get('channel_id')
    print(channel_id)
    #tokenize the user input
    new_user_input_ids = tokenizer.encode(input['text'] + tokenizer.eos_token, return_tensors='pt')
    #iterate through the chat_hist array
    for obj in chat_hist:
        #check if there's a history for that channel
        if obj['id'] == channel_id:
            #check the size of the hist, if it's over 'max_lenght' remove the oldest entry

            #add input to chat history
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
            #generate response from what the user asked
            #taking into account the context (chat history)
            chat_history_ids = model.generate(bot_input_ids, max_length=500,pad_token_id=
                tokenizer.eos_token_id,
                no_repeat_ngram_size=args.get('no_repeat_ngram_size'),
                do_sample=args.get('do_sample'),
                top_k=args.get('top_k'),
                top_p=args.get('top_p'),
                temperature=args.get('temperature')
                )
            #format the response into human readable stuff
            bot_response = "{}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))
            #store teh reply
            reply = bot_response
            #return the reply
            return reply
        else:
            #create that channels conversation history

            #add input to chat history
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
            #generate response from what the user asked
            #taking into account the context (chat history)
            chat_history_ids = model.generate(bot_input_ids, max_length=500,pad_token_id=
                tokenizer.eos_token_id,
                no_repeat_ngram_size=args.get('no_repeat_ngram_size'),
                do_sample=args.get('do_sample'),
                top_k=args.get('top_k'),
                top_p=args.get('top_p'),
                temperature=args.get('temperature')
                )
            #format the response into human readable stuff
            bot_response = "{}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))
            #store teh reply
            reply = bot_response
            #return the reply
            return reply

#define train function - used by ShanghAI to train the model
##decorator binds a piece of code to an URL path
##in this case, we link the /train path to the train() function
@post('/train', method='POST')
def train():
    return "blep"

#run the server on the host:port combination given
run(host='localhost', port=8000, debug=True)
