from wordcloud import WordCloud
from io import BytesIO
import base64
import pandas as pd
csv_filename_teacher_form ='teacher_evaluation.csv'
csv_filename_feedback_records = 'feedback_data.csv'



def generate_wordcloud():
    df_feedback = pd.read_csv('feedback_data.csv')
    df_evaluations = pd.read_csv('teacher_evaluations.csv')

    feedback_columns = ['additionalRemarks', 'BestFeatures', 'ImprovementAreas']
    evaluations_columns = ['additionalRemarks', 'BestFeatures', 'ImprovementAreas']

    feedback_text = df_feedback[feedback_columns].dropna().values.flatten()
    evaluations_text = df_evaluations[evaluations_columns].dropna().values.flatten()

    all_text = pd.concat([pd.Series(feedback_text), pd.Series(evaluations_text)])

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_text))

    image_stream = BytesIO()
    wordcloud.to_image().save(image_stream, format='PNG')
    image_stream.seek(0)
    encoded_image = base64.b64encode(image_stream.getvalue()).decode('utf-8')
    return encoded_image