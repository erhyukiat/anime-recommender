# App URL
[Hosted on Streamlit](https://share.streamlit.io/erhyukiat/anime-recommender/main/anime_recommender.py)

# Problem Statement
Anime, derived from the word "animation", refers to hand-drawn and computer animations from Japan. Prior to streaming services, it was difficult to watch anime outside of Japan. Then came fansubbers, who would add English subtitles to anime episodes and share them through illegal streaming sites or peer to peer file sharing. While this created an avenue for existing anime fans to watch their favourite animes, it does not help bring in new fans, as non-fans/prospective fans will only be exposed to anime through word of mouth.

Netflix added anime in 2014. From Google trends, there was an increase in interest of "anime" thereafter. Having anime on Netflix gave accessiblity and visiblity of animes. There is another increase in interest that coincides with the COVID-19 pandemic, likely explained by people staying safe at home and getting anime recommendations by Netflix.

However, Netflix's anime selections are limited to well-known titles like Demon Slayer/Kimetsu no Yaiba, Attack on Titan/Shingeki no Kyojin. The goal of this project is to develop a recommender system, hosted on a webapp, that recommends animes that are popular or have been rated highly on the anime database, Anilist. These animes may or may not be available in Netflix, the main goal here is to let people know what else is out there (as opposed to being limited to Netflix selections).

# Data
Anime and user data retrieved from [AniList](https://anilist.co/) [API](https://anilist.gitbook.io/anilist-apiv2-docs/) via GraphQL <br>

# Model: LightFM
Model and some parts of the code from [LightFM](https://github.com/lyst/lightfm) and [Microsoft Recommenders](https://github.com/microsoft/recommenders)

# Item Features
Genres of the anime<br>
Present genres: Action, Adventure, Comedy, Drama, Ecchi, Fantasy, Horror, Mahou Shoujo, Mecha, Music, Mystery, Psychological, Romance, Sci-Fi, Slice of Life, Sports, Supernatural, Thriller

# User Features
User's favourite genres
- Defined as the genres of the anime that the user has rated greater than or equal to their average/75th percentile rating

6 variants of user features were identified:<br>
1: All genres that the user likes (>= average) <br>
2: All genres that the user likes (>= 75th percentile) <br>
3: 3 most common genres among liked genres (>= average) <br>
4: 5 most common genres among liked genres (>= average) <br>
5: 3 most common genres among liked genres (>= 75th percentile) <br>
6: 5 most common genres among liked genres (>= 75th percentile)

# Model Scores
|                                              | Train AUC | Test AUC | Train Precision | Test Precision | Train Recall | Test Recall |
|----------------------------------------------|-----------|----------|-----------------|----------------|--------------|-------------|
| Baseline (Without user or item features)     | 0.581     | 0.554    | 0.001           | 0.0003         | 0.001        | 0.001       |
| Item Features & User Features Type 1         | 0.953     | 0.949    | 0.219           | 0.078          | 0.071        | 0.066       |
| Item Features & User Features Type 2         | 0.953     | 0.948    | 0.223           | 0.080          | 0.077        | 0.073       |
| Item Features & User Features Type 3         | 0.954     | 0.949    | 0.238           | 0.084          | 0.097        | 0.083       |
| Item Features & User Features Type 4         | 0.955     | 0.950    | 0.282           | 0.100          | 0.117        | 0.103       |
| Item Features & User Features Type 5         | 0.953     | 0.948    | 0.256           | 0.090          | 0.100        | 0.086       |
| Item Features & User Features Type 6         | 0.953     | 0.949    | 0.245           | 0.087          | 0.093        | 0.081       |
| Item Features & User Features Type 4 (tuned) | 0.959     | 0.952    | 0.340           | 0.122          | 0.138        | 0.120       |

Tuned model with item features and user feature type 4 as it yielded the best scores. Final model performed substantially better than the baseline model.

# Conclusion
The LightFM model, when given user and item features, performs really well as compared to the baseline model (which did not have any user or item features).

In recommending animes for new users, it seems the model is predicting popular animes which may or may not be relevant to the user input. E.g. "Romance" is part of the user input, but "Romance" only appears in 5 out of the 10 recommendations. Further to that, there are recommendations that do not even include one of the genres in the user input, e.g. Shingeki no Kyojin and Sword Art Online II does not have "Comedy", "Romance", nor "Slice of Life" genres. There is however, 1 recommendation that hits all user inputs, Kaguya-sama ga Kokurasetai. This means that 1 out of 10 recommendations are relevant, which coincides with the model's precision@k score of 12.2%.

In recommending animes for existing users or similar animes, the model is performing as desired, as the top 10 recommendations contain animes that are not available in Netflix at this moment.

Overall, the model is performing decently and the issue with recommending animes for new users might be attributed to a cold start problem, seeing how recommendations for existing users are working as desired. There may be merit in applying an additional filter to remove irrelevant recommendations based on user input, such as excluding animes that does not match any of the genres or does not match with all genres. On that note, it might be useful to allow the user to select if they want the recommendations to contain all or at least one of the genres they have selected.
