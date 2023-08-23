import numpy as np
import pandas as pd
import re


def prepare_data(conversations_files, conversations = {}):
     for conversation_person, conversation_data in conversations_files.items():
          messages = []

          with open(f'./data/{conversation_data}', encoding="utf-8") as f:
               conversation_messages = f.read()
               conversation_messages = conversation_messages.split('\n')

               for value in [re.findall(r'\[(.*?)\]\s*(.*?):\s*(.*)', conversation) for conversation in conversation_messages]:
                    if value:
                         messages.append(value[0])

          conversations[conversation_person] = messages

     return conversations


def create_dataframe(conversations, host, dataframes={}):
    for key, value in conversations.items():

        interactions = {key: [], host: []}
        current_person = ''
        for date, name, content in value:
            if name in interactions:
                if name != current_person:
                    if interactions[key] and len(interactions[key]) != len(interactions[host]):
                        diff = len(interactions[host]) - len(interactions[key])
                        interactions[key] += [''] * diff
                    current_person = name
                    interactions[name].append('')
                interactions[name][-1] += ' ' + content.strip()

        # Completar con espacios en blanco si los largos no coinciden
        if len(interactions[key]) > len(interactions[host]):
            diff = len(interactions[key]) - len(interactions[host])
            interactions[host].append([''] * diff)
        elif len(interactions[key]) < len(interactions[host]):
            diff = len(interactions[host]) - len(interactions[key])
            interactions[key].append([''] * diff)
        
        dataframes[key] = pd.DataFrame(interactions)
    
    df_final = pd.DataFrame(columns=["user", "prompt"])

    for data in dataframes.values():
        data.columns = ["user", "prompt"]

        df_final = pd.concat([df_final, data])
    
    return df_final


def clean_dataframe(df_final):
    cases_to_replace = [
    "image omitted",
    "video omitted",
    "audio omitted",
    "sticker omitted",
    "Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them.",
    "['']",
    "\u200e"
    ]
    for case in cases_to_replace:
        df_final["user"] = df_final["user"].apply(lambda x: str(x).strip().replace(case, ''))
        df_final["prompt"] = df_final["prompt"].apply(lambda x: str(x).strip().replace(case, ''))

    df_final["user"] = df_final["user"].apply(lambda x: x.lower())
    df_final["prompt"] = df_final["prompt"].apply(lambda x: x.lower())

    df_final.replace('', np.nan, inplace=True)
    df_final.replace('\u200e', np.nan, inplace=True)


    df_final = df_final.dropna()

    return df_final


def create_tuples_text(df_final):
    tuples_list = [(row[0], row[1]) for row in df_final.values]
    
    return tuples_list


# YOUR CODE STARTS HERE
host = "Daniel"
conversations_files = {"Daniel Roca EY": "_chat_Dani_Roca.txt", "Diego Chaparro UNAL": "_chat_Diego.txt",
                        "Juanes Cepeda Primo" :"_chat_Juanes.txt",
                         "Ma" :"_chat_ma.txt", "Pa": "_chat_pa.txt", "Sebas Col Puerto": "_chat_Sebas.txt"}
total_messages = 50
instruction = """ 
La siguiente es una lista de tuplas que representa la conversación entre dos usuarios.
Cada tupla consta de dos elementos donde el primero de cada una contiene el texto
del usuario 1 y el otro elemento la respuesta del usuario 2 a aquello dicho por el usuario 1:
"""
prompt_engineering = """
A partir de ahora eres un chatbot que habla con el mismo estilo de conversación que el usuario 2,
otorgando respuestas cortas tal como el usuario 2 lo haría.
"""



if __name__ == '__main__':
    conversations = prepare_data(conversations_files)
    df_final = create_dataframe(conversations, host="Daniel")
    df_final = clean_dataframe(df_final)
    df_final.to_csv("./data/data_prompts.csv", index=False)
    tuples_list = create_tuples_text(df_final)

    # content = (instruction + ' ' + str(tuples_list[:total_messages]) +  ' ' + prompt_engineering).replace("\n", '')
    content = (instruction + "\n\n" + str(tuples_list[:total_messages]) +  "\n\n" + prompt_engineering)

    with open('./data/output.txt', 'w') as file:
        # Write each tuple to the file
        file.write(content)