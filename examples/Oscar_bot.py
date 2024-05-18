from openai import OpenAI
import tiktoken
from scipy import spatial 
import pandas as pd

client = OpenAI(
    api_key=""
)

df=pd.read_csv('./data/oscars.csv')
print(df.head())

df=df.loc[df['year_ceremony'] == 2023]
df=df.dropna(subset=['film'])
df['category'] = df['category'].str.lower()
df.head()

df['text'] = df['name'] + ' got nominated under the category, ' + df['category'] + ', for the film ' + df['film'] + ' to win the award'
df.loc[df['winner'] == False, 'text'] = df['name'] + ' got nominated under the category, ' + df['category'] + ', for the film ' + df['film'] + ' but did not win'
print(df.head()['text'])

def text_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    #return response["data"][0]["embedding"]
    embedding = response.data[0].embedding
    return embedding


df=df.assign(embedding=(df["text"].apply(lambda x : text_embedding(x))))
print(df.head())

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    
    EMBEDDING_MODEL = "text-embedding-ada-002"
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

strings, relatednesses = strings_ranked_by_relatedness("Lady Gaga", df, top_n=3)
for string, relatedness in zip(strings, relatednesses):
    print(f"{relatedness=:.3f}")
    print(string)
    
def num_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))    

def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below content related to 95th Oscar awards to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_row = f'\n\nOscar database section:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_row + question)
            > token_budget
        ):
            break
        else:
            message += next_row
    return message + question

def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = "gpt-3.5-turbo",
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about 95th Oscar awards."},
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    return response_message

print(ask('What was the nomination from Lady Gaga for the 95th Oscars?'))  
print(ask('What were the nominations for the music awards?'))