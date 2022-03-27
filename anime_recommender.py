import pickle
import re
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import sparse
from recommenders.models.lightfm.lightfm_utils import similar_items

anime_titles_list = pickle.load(open('./assets/anime_titles.pkl', 'rb'))
anime_genres_list = pickle.load(open('./assets/anime_genres.pkl', 'rb'))
model = pickle.load(open('./assets/model.pkl', 'rb'))
data_items_for_predictions = pickle.load(open('./assets/data_items_for_predictions.pkl', 'rb'))
item_features = data_items_for_predictions['item_features']  
uid_map = data_items_for_predictions['uid_map']
iid_map = data_items_for_predictions['iid_map']
iid_map_reverse = data_items_for_predictions['iid_map_reverse']
ufeature_map = data_items_for_predictions['ufeature_map']
n_items = data_items_for_predictions['n_items']
unique_media_id = data_items_for_predictions['unique_media_id']
anime_db = pd.read_pickle('./assets/anime_db_export.pkl')

def make_predictions_with_genres(genres, num_preds, genres_pred_type):
    # manipulate data into a format that we pass to our model
    new_user_features = format_newuser_input(ufeature_map, genres)
    model_pred = model.predict(0, np.arange(n_items), user_features=new_user_features)
    pred_dict = {'media_id':unique_media_id,'pred':model_pred}
    pred_df = pd.DataFrame(pred_dict)
    pred_df = pd.merge(pred_df,anime_db,on='media_id')
    pred_df['genres_list'] = pred_df['genres'].apply(str.split,args=(", ",))
    pred_df['genres_found'] = pred_df['genres_list'].apply(all_genres_found, args=(genres,genres_pred_type,))
    output_df = pred_df[pred_df['genres_found']].sort_values('pred',ascending=False).head(num_preds)
    output_df.reset_index(inplace=True,drop=True)
    output_df.drop(columns=['media_id','pred','coverImage_medium'],inplace=True)
    output_dict = output_df.to_dict('records')
    
    # return the list of dicts containing recommendations
    return output_dict

def make_predictions_with_titles(titles, num_preds):
    titles_id = []
    
    for title in titles:
        titles_id.extend(anime_db[anime_db['title_romaji']==title]['media_id'].values)
        titles_id.extend(anime_db[anime_db['title_english']==title]['media_id'].values)
    titles_id = list(set(titles_id))
    
    similar_pred_all = pd.DataFrame()

    for title_id in titles_id:
        item_x = iid_map[title_id]
        similar_pred = similar_items(item_id = item_x, item_features=item_features, model=model,N=num_preds*2)
        similar_pred['media_id'] = similar_pred['itemID'].map(iid_map_reverse)
        similar_pred_all = pd.concat([similar_pred_all,similar_pred[['media_id','score']]],ignore_index=True)
    similar_pred_all = similar_pred_all[~similar_pred_all['media_id'].isin(titles_id)]
    similar_pred_all.drop_duplicates(subset=['media_id'],inplace=True)
    output_pred = similar_pred_all.sort_values('score',ascending=False).head(num_preds)
    output_pred = pd.merge(output_pred,anime_db,on='media_id')
    output_pred.drop(columns=['media_id','score','coverImage_medium'],inplace=True)
    output_dict = output_pred.to_dict('records')    

    # return the results template with our prediction value filled in
    return output_dict

def format_newuser_input(user_feature_map, user_feature_list):
    #user_feature_map = user_feature_map  
    num_features = len(user_feature_list)
    normalised_val = 1.0 
    target_indices = []
    for feature in user_feature_list:
        try:
            target_indices.append(user_feature_map[feature])
        except KeyError:
            print(f"new user feature encountered '{feature}'")
            pass
    #print("target indices: {}".format(target_indices))
    new_user_features = np.zeros(len(user_feature_map.keys()))
    for i in target_indices:
        new_user_features[i] = normalised_val
    new_user_features = sparse.csr_matrix(new_user_features)
    return(new_user_features)

def all_genres_found(genres, user_input_genres, pred_type):
    input_genres_found = [False] * len(user_input_genres)
    
    for genre in genres:
        for idx,input_genre in enumerate(user_input_genres):
            if genre == input_genre:
                input_genres_found[idx] = True
    
    if pred_type == 'all':
        return False if False in input_genres_found else True
    elif pred_type =='at_least_one':
        return True if True in input_genres_found else False
    
    return False

st.title("Anime recommender")
st.write("By Erh Yu Kiat [GitHub](https://github.com/erhyukiat)")
st.markdown("[Source code](https://github.com/erhyukiat/anime-recommender) | Model from [LightFM](https://github.com/lyst/lightfm) and [Microsoft Recommenders](https://github.com/microsoft/recommenders)")
st.text("")
st.text("")

input_type = st.radio("Select your input type:", ('Favourite genres','Favourite anime titles'))
#input_type = st.radio("Select your input type:", ('Favourite genres','Favourite anime titles','Anilist user ID'))

if  input_type == "Favourite genres":
    container = st.container()
    all = st.checkbox("Select all")

    if all:
        genres = container.multiselect("Select your favourite genres:", anime_genres_list, anime_genres_list)
        genres_prediction_type = st.radio('Do you want your recommendations to match at least one of your selections or to have all selected genres?',
                                     ['Match at least one of the selected genres'])
    else:
        genres =  container.multiselect("Select your favourite genres:", anime_genres_list)
        genres_prediction_type = st.radio('Do you want your recommendations to match at least one of your selections or to have all selected genres?',
                                     ('Match at least one of the selected genres','Have all selected genres'))
    #genres = st.multiselect('Select your favourite genres:',anime_genres_list)
    
elif input_type == "Favourite anime titles":
    titles = st.multiselect('Select your favourite anime titles:',anime_titles_list)
elif input_type == "Anilist user ID":
    st.caption('Disclaimer: Your user ID may not be found in the data as only a small subset of Anilist users are used to train the model')
    user_id = st.text_input('Enter your Anilist user ID')
    
num_preds = st.slider("How many recommendations do you want?", 1, 100, 10)

if st.button("Recommend"):
    if input_type == "Favourite genres":
        if genres_prediction_type == 'Have all selected genres':
            genres_pred_type = 'all'
        elif genres_prediction_type == 'Match at least one of the selected genres':
            genres_pred_type = 'at_least_one'
        
        if len(genres) > 0:
            recommendations = make_predictions_with_genres(genres, num_preds, genres_pred_type)
            count = 0

            for rec in recommendations:
                count += 1

                st.header(f"Recommendation #{count}")

                # title
                st.subheader(f"{rec['title_romaji']}")
                if rec['title_romaji'] != rec['title_english']:
                    # Show english title if english and romaji title are not the same
                    st.write(f"Alternative title: {rec['title_english']} ")

                image_col, details_col = st.columns([2,3])

                with image_col:
                    # image
                    st.image(f"{rec['coverImage_large']}")

                with details_col:
                    # description                
                    st.write(f"Description: {re.sub('<[^<]+?>', '', rec['description'].strip())}")
                    # averageScore
                    st.write(f"Average score: {rec['averageScore']}")
                    # genres
                    st.write(f"Genres: {rec['genres']}")
                    # siteUrl
                    st.write(f"Anilist URL: {rec['siteUrl']}")
                    # startDate
                    if pd.isnull(rec['startDate']):
                        st.write(f"Start date: -")
                    else:
                        st.write(f"Start date: {rec['startDate'].strftime('%d %b %Y')}")
                        #st.write(f"Start date: {datetime.strptime(rec['startDate'],'%a, %d %b %Y %H:%M:%S GMT').strftime('%d %b %Y')}")
                    # endDate
                    if pd.isnull(rec['endDate']):
                        st.write(f"End date: -")
                    else:
                        st.write(f"End date: {rec['endDate'].strftime('%d %b %Y')}")
                        #st.write(f"End date: {datetime.strptime(rec['endDate'],'%a, %d %b %Y %H:%M:%S GMT').strftime('%d %b %Y')}")
                    # duration
                    if pd.isnull(rec['duration']):
                        st.write(f"Episode duration: -")
                    else:
                        st.write(f"Episode duration: {int(rec['duration'])} mins")
        else:
            st.write("No genres selected")
                
        st.text("")

    elif input_type == "Favourite anime titles":
        if len(titles) > 0 and (num_preds>=1 and num_preds <= 100):
            recommendations = make_predictions_with_titles(titles, num_preds)
            count = 0

            for rec in recommendations:
                count += 1
                st.header(f"Recommendation #{count}")

                # title
                st.subheader(f"{rec['title_romaji']}")
                if rec['title_romaji'] != rec['title_english']:
                    # Show english title if english and romaji title are not the same
                    st.write(f"Alternative title: {rec['title_english']} ")

                image_col, details_col = st.columns([2,3])

                with image_col:
                    # image
                    st.image(f"{rec['coverImage_large']}")

                with details_col:
                    # description
                    st.write(f"Description: {re.sub('<[^<]+?>', '', rec['description'].strip())}")
                    # averageScore
                    st.write(f"Average score: {rec['averageScore']}")
                    # genres
                    st.write(f"Genres: {rec['genres']}")
                    # siteUrl
                    st.write(f"Anilist URL: {rec['siteUrl']}")
                    # startDate
                    if pd.isnull(rec['startDate']):
                        st.write(f"Start date: -")
                    else:
                        st.write(f"Start date: {rec['startDate'].strftime('%d %b %Y')}")
                        #st.write(f"Start date: {datetime.strptime(rec['startDate'],'%a, %d %b %Y %H:%M:%S GMT').strftime('%d %b %Y')}")
                    # endDate
                    if pd.isnull(rec['endDate']):
                        st.write(f"End date: -")
                    else:
                        st.write(f"End date: {rec['endDate'].strftime('%d %b %Y')}")
                        #st.write(f"End date: {datetime.strptime(rec['endDate'],'%a, %d %b %Y %H:%M:%S GMT').strftime('%d %b %Y')}")
                    # duration
                    if pd.isnull(rec['duration']):
                        st.write(f"Episode duration: -")
                    else:
                        st.write(f"Episode duration: {int(rec['duration'])} mins")
        else:
            st.write("No titles selected")
        st.text("")

    elif input_type == "Existing user":
        pass
    else:
        st.write("Invalid input type")