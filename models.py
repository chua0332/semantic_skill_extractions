from ssg_sea.extract_skills import extract_skills, batch_extract_skills
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from numpy import load
import requests
import os



'''
dir = path.Path(__file__).abspath()
sys.path.append(dir.parent.parent)
'''

#initializing language model
#model = SentenceTransformer('all-mpnet-base-v2')

API_TOKEN = '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'
API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-mpnet-base-v2"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	

#intitializing JINZHA span extraction model
token_skill_classifier = pipeline(model="jjzha/jobbert_skill_extraction", aggregation_strategy="first")
token_knowledge_classifier = pipeline(model="jjzha/jobbert_knowledge_extraction", aggregation_strategy="first")

#Importing the saved MPNET's embeddings
embeddings = load(os.path.join(os.getcwd(),'skills_embeddings.npy'))


#Reading in the deduped skills titles
df = pd.read_csv(os.path.join(os.getcwd(),'skill_master_dedup_06nov2022.csv'))
df = df[['skill_id', 'skill_title', 'dup_parent']]
df['merged_title'] = df['dup_parent'].combine_first(df.skill_title)
df['source'] = 'skill title'

df = df[['merged_title', 'skill_title', 'skill_id', 'source']]

df2 = pd.read_csv(os.path.join(os.getcwd(),'unique_2k_skills.csv'))
df2 = df2[['skill_title','skill_description_final','skill_id']]
df2['source'] = 'full_description'
df2.columns = ['merged_title', 'skill_title','skill_id','source']

df_cleaned = pd.concat([df,df2], ignore_index=True)
df_cleaned = df_cleaned.dropna()


#Function to aggregate single words to next span if it is adjacent --> part of JINZHA original script
def aggregate_span(results):
    new_results = []
    current_result = results[0]

    for result in results[1:]:
        if result["start"] == current_result["end"] + 1:
            current_result["word"] += " " + result["word"]
            current_result["end"] = result["end"]
        else:
            new_results.append(current_result)
            current_result = result

    new_results.append(current_result)

    return new_results

df_clean = pd.DataFrame(list(zip(embeddings, list(df_cleaned['merged_title']), list(df_cleaned['source']))))

def find_similar(q,k):
    testing = query({"inputs":q})
    testing = np.array(testing)
    trial = []
    testing = testing.reshape(1,-1)
    vals = cosine_similarity(testing,embeddings)
    idx_asc = vals.argsort()[0][-k:]
    idx_dsc = idx_asc[::-1]
    flatv = np.sort(vals[0])
    vk_asc = flatv[-k:]
    vk_dsc = vk_asc[::-1]
    if(vk_dsc[0]==0):
      print("No skills matched")
    else:
      for v, i in zip(vk_dsc, idx_dsc):
        a = {'score' : float(v), 'skill' : df_clean[1][i], 'phrase' : q, 'matched_by': df_clean[2][i]}
        trial.append(a)
      df_output = pd.DataFrame(trial)
      return df_output
    
#Function for combined list output (combining tools/skills for now) --> for MPNET with span
def ner_combined_latest(text: str):
    output = []
    
    
    for line in text.split('  '):
        if len(line) == 0:
            continue
        
        b = token_skill_classifier(line)
        c = token_knowledge_classifier(line)
        output += b + c

    if len(output) > 0:
        output_skills = aggregate_span(output)
    skill_list = []
    for i in output_skills:
        skill_list.append(i['word'])

    df_skills_extracted = pd.DataFrame(columns = ['score','skill','phrase', 'matched_by'])

    #mpnet portion with span extraction
    for i in range(len(skill_list)):
        df_skills_extracted = pd.concat([find_similar(skill_list[i], 1), df_skills_extracted], ignore_index = True)

    span_extracted = df_skills_extracted.sort_values('score', ascending = False).drop_duplicates(subset ='skill', keep = 'first')

    span_ex_list = list(span_extracted.loc[span_extracted['score'] >= 0.55]['skill'].sort_values())

    return span_ex_list

# Pure mpnet extraction
def mpnet_extract(text: str): 
    if len(text.split())<=512:
        text = text[:512]
        ex_list = find_similar(text, 20)
        ex_list = ex_list.drop_duplicates(subset = 'skill', keep = 'first')
        final_extract_list = list(ex_list.loc[ex_list['score'] >= 0.55]['skill'].sort_values())
    else:
        text = text[:512]
        text2 = text[512:]
        ex_list = find_similar(text, 20)
        ex_list = ex_list.drop_duplicates(subset = 'skill', keep = 'first')
        ex_final = list(ex_list.loc[ex_list['score'] >= 0.55]['skill'].sort_values())
        ex_list2 = find_similar(text2, 20)
        ex_list2 = ex_list2.drop_duplicates(subset = 'skill', keep = 'first')
        ex_final2 = list(ex_list2.loc[ex_list2['score'] >= 0.55]['skill'].sort_values())
        final_extract_list = ex_final + ex_final2
        
    return final_extract_list

def mpnet_combined(text: str):
    a = ner_combined_latest(text)
    b = mpnet_extract(text)
    #Combining both span extraction + mpnet sentence embeddings
    mpnet_combinelist = a + b
    #Converting to lower case so i can remove for duplications
    mpnet_combinelist = [x.lower() for x in mpnet_combinelist]
    mpnet_combinelist = list(dict.fromkeys(mpnet_combinelist))
    mpnet_combinelist.sort()
    
    return mpnet_combinelist
    
    

#For SEA V1
def sea_extract_skills(text: str):
    sea_v1_list = []
    for item in extract_skills(text).values():
        for skills in item.values():
            sea_v1_list.append(skills['skill_title'])
    
    return sea_v1_list

'''
#For ADA002
def ada_extract_skills(text: str):
    ada_skills_list = []
    r = requests.post('https://ssg-course-search-ai.herokuapp.com/skills_finder', json={'query': text})
    r = r.json()
    
    for item in r['matches']:
        if item['score'] >= 0.8:
            ada_skills_list.append(item['metadata']['skill_title'])
    return ada_skills_list

'''

#Skills Extractions Overall

def main_skills_extractor(text: str, flag):
    if flag == 'SEAv1':
        extractedskills_list = sea_extract_skills(text)
        
    if flag == 'SEAv1_plus':
        extractedskills_list = mpnet_combined(text)
    
    return extractedskills_list
  