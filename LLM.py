import csv
import json
import pandas as pd
from pathlib import Path
import os
from openai import OpenAI

# @author Kelly Yap (kellyy)
# @course CS 5664 Spring 2026
# notes: must configure your API key in environment variables!

# Config for LLM access
def api():
    CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
    MODEL = "llama3.1-8b"
    client = OpenAI(
        api_key=CEREBRAS_API_KEY,
        base_url="https://api.cerebras.ai/v1"
    )

    return client, MODEL

# Returns dataframe of 1k posts
def load():
    DATA = Path("dataset")
    df = pd.read_csv(DATA / "Truth_Seeker_Model_Dataset.csv")

    # Sample 150 balanced posts (fixed seed of 42, for testing only)
    # true = df[df["target"] == True].sample(75, random_state=42)
    # false = df[df["target"] == False].sample(75, random_state=42)
    # sample_df = pd.concat([true, false]).sample(frac=1, random_state=42)
    # sample_df = sample_df.reset_index(drop=True)

    # Unbalanced sample
    sample_df = df.sample(1000, random_state=42)

    return sample_df

# Conducts LLM prompting and returns the prediction
def prompt(client, model, post_text):

    SYSTEM_PROMPT = (
        "You are an expert at identifying fake news in posts on the platform Twitter/X. "
        "Classify the post as True or False. "
        "Respond ONLY with 1 valid JSON and no other text. Do not write an introduction or summary.: "
        '{"label": "True" or "False", '
        '"explanation": "2-3 sentences justifying your decision."}'
    )
    valid_schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "explanation": {"type": "string"},
        },
        "required": ["label", "explanation"],
        "additionalProperties": False
    }
    user_msg = f'Now classify this post:'
    user_msg += f'Post: "{post_text[:500]}"\nOutput:'

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ],
        response_format={
        "type": "json_schema", 
        "json_schema": {
            "name": "valid_schema",
            "strict": True,
            "schema": valid_schema
            }
        },
        temperature=0.0,
        max_tokens=256,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response")
    try:
        prediction = json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(f"Bad JSON: {content}")
    return prediction

# Calculates statistics of LLM prompting given lists of TP, FP, FN, TN
def eval(tp, fp, fn, tn):
    numTP = len(tp)
    numFP = len(fp)
    numFN = len(fn)
    numTN = len(tn)

    print("Accuracy: " + str((numTP + numTN) / (numTP + numFP + numFN + numTN)))

    falsePrecision = numTP / (numTP + numFP)
    truePrecision = numTN / (numTN + numFN)
    macroPrecision = (falsePrecision + truePrecision) / 2
    print("Precision (macro): " + str(macroPrecision))

    falseRecall = numTP / (numTP + numFN)
    trueRecall = numTN / (numTN + numFP)
    macroRecall = (falseRecall + trueRecall) / 2
    print("Recall (macro): " + str(macroRecall))

    print("F1-Score (macro): " + str(2 * ((macroPrecision * macroRecall) / (macroPrecision + macroRecall))))

    print("# True Positives: " + str(len(tp)))
    print("# False Positives: " + str(len(fp)))
    print("# False Negatives: " + str(len(fn)))
    print("# True Negatives: " + str(len(tn)))

    print("Correct prediction of Fake (TP):")
    if (numTP != 0):
        print("Post: " + tp[0][1])
        print("Reasoning: " + tp[0][0]["explanation"] + "\n")
    else:
        print("None")

    print("Correct prediction of True (TN):")
    if (numTN != 0):
        print("Post: " + tn[0][1])
        print("Reasoning: " + tn[0][0]["explanation"] + "\n")
    else:
        print("None")

    print("Incorrect prediction of Fake (FP):")
    if (numFP != 0):
        print("Post: " + fp[0][1])
        print("Reasoning: " + fp[0][0]["explanation"] + "\n")
    else:
        print("None")

    print("Incorrect prediction of True (FN):")
    if (numFN != 0):
        print("Post: " + fn[0][1])
        print("Reasoning: " + fn[0][0]["explanation"] + "\n")
    else:
        print("None")

# Conducts prompting and evaluation
def llmMethod(posts, client, model):
    tp = [] # True postives, predicted False and is False
    fp = [] # False positives, predicted False but is True
    fn = [] # False negatives, predicted True but is False
    tn = [] # True negatives, predicted True and is True

    progress = 1 # for testing

    for post in posts.itertuples(index=False):
        # prediction = prompt(client, model, post.tweet)

        # if prediction["label"] == "False" and post.target == False: # TP
        #     tp.append((prediction, post.tweet[:500]))
        # elif prediction["label"] == "False" and post.target == True: # FP
        #     fp.append((prediction, post.tweet[:500]))
        # elif prediction["label"] == "True" and post.target == False: # FN
        #     fn.append((prediction, post.tweet[:500]))
        # else: # TN
        #     tn.append((prediction, post.tweet[:500]))

        try:
            prediction = prompt(client, model, post.tweet)

            if prediction["label"] == "False" and post.target == False:
                tp.append((prediction, post.tweet[:500]))
            elif prediction["label"] == "False" and post.target == True:
                fp.append((prediction, post.tweet[:500]))
            elif prediction["label"] == "True" and post.target == False:
                fn.append((prediction, post.tweet[:500]))
            else:
                tn.append((prediction, post.tweet[:500]))

        except Exception as e:
            print("\n--- ERROR ---")
            print("Post:", post.tweet[:500])
            print("Error:", str(e))
            print("--------------\n")
            continue

        print(f"Processed: {progress}/{len(posts)}", end="\r", flush=True)
        progress += 1

    print()
    eval(tp, fp, fn, tn)

def main():
    posts = load()
    client, model = api()
   
    print("LLM Prompting")
    llmMethod(posts, client, model)

if __name__ == "__main__":
    main()