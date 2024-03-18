
import plotly.express as px
import os
import matplotlib
from flask import Flask, request, render_template, session, redirect, url_for, flash
from Student_csv import  update_enrollment_status, read_courses_from_csv, students_courses_file, Student, add_student_to_csv, update_student_in_csv, delete_student_from_csv,csv_filename_students
import csv
from course_csv import add_course_to_csv, update_course_in_csv, delete_course_from_csv, csv_filename_courses
from Course_add import csv_filename_assignment_records, get_available_courses,assign_course_to_teacher
from subjects_csv import add_subject_to_csv, update_subject_in_csv, delete_subject_from_csv,csv_filename_subjects
from faculty_csv import Admin,Teacher, add_faculty_to_csv, update_faculty_in_csv, delete_faculty_from_csv,csv_filename_faculty
from datetime import datetime
from flask import jsonify
from feedback_file import course_data, csv_filename_feedback_records
from teacher_evaluation import get_teacher_details, check_attendance,write_to_csv
import re
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
matplotlib.use('Agg')  # Set the backend to Agg


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


app = Flask(__name__)
app.secret_key = os.urandom(12)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def check_csv_exists(filename):
    return os.path.exists(filename)

def create_csv(filename, headers):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

@app.route('/')
def index():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        if not check_csv_exists(csv_filename_students):
            create_csv(csv_filename_students, ['student_username', 'password'])
        if not check_csv_exists(csv_filename_courses):
            create_csv(csv_filename_courses, ['Course ID','course_name'])
        if not check_csv_exists(csv_filename_subjects):
            create_csv(csv_filename_subjects, ['Subject Name'])
        if not check_csv_exists(csv_filename_faculty):
            create_csv(csv_filename_faculty, ['username','password'])
        if not check_csv_exists(students_courses_file):
            create_csv(students_courses_file, ['students_courses'])
        if not check_csv_exists(students_courses_file):
            create_csv(students_courses_file,['students_courses'])
        if not check_csv_exists(csv_filename_assignment_records):
            create_csv(csv_filename_assignment_records,['username','Course ID','course_name'])
        available_courses = get_available_courses()
        teachers = Teacher.read_csv(csv_filename_faculty)
        aspects = [
            'Learning Outcomes', 'Workload Manageable', 'Well Organized',
            'Student Participation', 'Student Progress',
            'Well Structured', 'Encouraged Participation',
            'Conducive to Learning', 'Learning Materials', 'Recommended Reading', 'Library Resources',
            'Online Resources',
            'Interest Stimulated', 'Course Pace', 'Clear Presentation',
            'Assessment Reasonable', 'Feedback Timely', 'Feedback Helpful'
        ]

    return render_template('admin.html', courses=available_courses, teachers=teachers,aspects=aspects)


@app.route('/sentiment_scores')
def view_sentiment_scores():
    df_feedback = pd.read_csv('feedback_data.csv')

    unique_course_ids = df_feedback['Course ID'].unique()

    course_info_dict = {}

    df_assignment_records = pd.read_csv('assignment_records.csv')

    for course_id in unique_course_ids:
        course_assignment_records = df_assignment_records[df_assignment_records['Course ID'] == course_id]

        if not course_assignment_records.empty:
            username = course_assignment_records.iloc[0]['username']
            course_name = course_assignment_records.iloc[0]['course_name']
            course_info_dict[course_id] = {'username': username, 'course_name': course_name}

    feedback_questions = ['Learning Outcomes', 'Workload Manageable', 'Well Organized',
                          'Student Participation', 'Student Progress',
                          'Well Structured', 'Encouraged Participation', 'Conducive to Learning',
                          'Learning Materials', 'Recommended Reading', 'Library Resources', 'Online Resources',
                          'Interest Stimulated', 'Course Pace', 'Clear Presentation',
                          'Assessment Reasonable', 'Feedback Timely', 'Feedback Helpful']

    def get_sentiment_category(score):
        if score < -1:
            return 'Very Negative'
        elif -1 <= score < 0:
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        elif 0 < score < 1:
            return 'Positive'
        else:
            return 'Very Positive'

    sentiment_categories = []

    def calculate_subjective_score(text):
        sentiment = TextBlob(str(text)).sentiment.polarity

        if sentiment > 0.2:
            return 1
        elif sentiment > 0:
            return 0.5
        elif sentiment == 0:
            return 0
        elif sentiment > -0.2:
            return -0.5
        else:
            return -1

    df_feedback['BestFeatures_Subjective_Score'] = df_feedback['BestFeatures'].apply(calculate_subjective_score)

    df_feedback['ImprovementAreas_Subjective_Score'] = df_feedback['ImprovementAreas'].apply(calculate_subjective_score)

    df_feedback.to_csv('feedback_data_with_subjective_scores.csv', index=False, mode='w')

    best_features_subjective_scores = []
    improvement_areas_subjective_scores = []

    for course_id, course_data in df_feedback.groupby('Course ID'):
        total_sentiment_score = 0
        for question in feedback_questions:
            total_sentiment_score += course_data[question].mean()
        average_sentiment_score = total_sentiment_score / len(feedback_questions)

        sentiment_category = get_sentiment_category(average_sentiment_score)

        sentiment_categories.append(sentiment_category)

        best_features_sentiment = TextBlob(course_data['BestFeatures'].to_string()).sentiment.polarity
        improvement_areas_sentiment = TextBlob(course_data['ImprovementAreas'].to_string()).sentiment.polarity

        best_features_subjective_scores.append(1 if best_features_sentiment > 0 else 0)
        improvement_areas_subjective_scores.append(1 if improvement_areas_sentiment > 0 else 0)

    while len(sentiment_categories) < len(unique_course_ids):
        sentiment_categories.append(None)
        best_features_subjective_scores.append(None)
        improvement_areas_subjective_scores.append(None)

    scores_df = pd.DataFrame({'Course ID': unique_course_ids,
                              'Sentiment Category': sentiment_categories,
                              'BestFeatures_Subjective_Score': best_features_subjective_scores,
                              'ImprovementAreas_Subjective_Score': improvement_areas_subjective_scores})

    for index, row in scores_df.iterrows():
        course_id = row['Course ID']
        if course_id in course_info_dict:
            scores_df.loc[index, 'username'] = course_info_dict[course_id]['username']
            scores_df.loc[index, 'course_name'] = course_info_dict[course_id]['course_name']
        else:
            scores_df.loc[index, 'username'] = 'Unknown'
            scores_df.loc[index, 'course_name'] = 'Unknown'

    return render_template('sentiment_scores.html', scores=scores_df)




@app.route('/view_data')
def view_data():
    df_feedback = pd.read_csv('feedback_data.csv')
    df_users_courses = pd.read_csv('assignment_records.csv')
    df_merged = pd.merge(df_feedback, df_users_courses, on='Course ID')
    df_merged = df_merged.drop_duplicates(subset='Course ID')

    columns_of_interest = ['Learning Outcomes', 'Workload Manageable', 'Well Organized',
                           'Student Participation', 'Student Progress',
                           'Well Structured', 'Encouraged Participation', 'Conducive to Learning',
                           'Learning Materials', 'Recommended Reading', 'Library Resources', 'Online Resources',
                           'Interest Stimulated', 'Course Pace', 'Clear Presentation',
                           'Assessment Reasonable', 'Feedback Timely', 'Feedback Helpful']

    df_merged[columns_of_interest] = df_merged[columns_of_interest].apply(pd.to_numeric, errors='coerce')

    course_info = df_merged.groupby('Course ID').agg({'username': lambda x: '<br>'.join(x),
                                                      'course_name': lambda x: '<br>'.join(x)})

    df_merged = pd.merge(df_merged[['Course ID'] + columns_of_interest].drop_duplicates(),
                         course_info,
                         on='Course ID',
                         how='left')

    plots = []
    for column in columns_of_interest:
        fig = px.bar(df_merged, x='Course ID', y=column,
                     title=f'Distribution of {column} by Course ID',
                     labels={'Course ID': 'Course ID', column: column})

        for i in range(len(df_merged)):
            usernames = df_merged['username'].iloc[i]
            course_names = df_merged['course_name'].iloc[i]
            fig.add_annotation(x=df_merged['Course ID'].iloc[i], y=df_merged[column].iloc[i],
                               text=f'Usernames: {usernames}<br>Course Names: {course_names}',
                               showarrow=False, font=dict(color='black', size=10), align='left')

        plots.append(fig.to_html(full_html=False))

    return render_template('view_data.html', plots=plots)


@app.route('/add_student', methods=['POST'])
def add_student():
    student_username = request.form['student_username']
    student_password = request.form['student_password']
    result = add_student_to_csv(student_username, student_password,csv_filename_students)
    return result

@app.route('/update_student', methods=['POST'])
def update_student():
    student_username = request.form['update_student_username']
    student_password = request.form['student_password']
    new_student_name = request.form['new_student_name']
    new_student_password = request.form['new_student_password']
    result = update_student_in_csv(student_username, student_password, new_student_name, new_student_password)
    return result

@app.route('/delete_student', methods=['POST'])
def delete_student():
    student_username = request.form['delete_student_username']
    result = delete_student_from_csv(student_username)
    return result

@app.route('/add_course', methods=['POST'])
def add_course():
    course_id = request.form['course_id']
    course_name = request.form['course_name']
    result = add_course_to_csv(course_id, course_name)
    return result

@app.route('/update_course', methods=['POST'])
def update_course():
    update_course_id = request.form['update_course_id']
    new_course_name = request.form['new_course_name']
    result = update_course_in_csv(update_course_id, new_course_name)
    return result

@app.route('/delete_course', methods=['POST'])
def delete_course():
    delete_course_id = request.form['delete_course_id']
    result = delete_course_from_csv(delete_course_id)
    return result

@app.route('/add_subject', methods=['POST'])
def add_subject():
    subject_name = request.form['subject_name']
    result = add_subject_to_csv(subject_name)
    return result

# Route for updating a subject
@app.route('/update_subject', methods=['POST'])
def update_subject():
    subject_name = request.form['subject_name']
    new_subject_name = request.form['new_subject_name']
    result = update_subject_in_csv(subject_name, new_subject_name)
    return result

# Route for deleting a subject
@app.route('/delete_subject', methods=['POST'])
def delete_subject():
    subject_name = request.form['delete_subject_name']
    result = delete_subject_from_csv(subject_name)
    return result

@app.route('/assign_course', methods=['POST'])
def assign_course():
    try:
        teacher_username = request.form['teacher']
        course_id = request.form['course_id']
        course_name = request.form['course_name']

        result = assign_course_to_teacher(teacher_username, course_id, course_name, csv_filename_faculty, csv_filename_assignment_records)
        return result
    except KeyError as e:
        return f"Error: Missing key in form data - {e}"
    except Exception as e:
        return f"Error: {e}"


@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Admin login attempt
        if Admin.login(username, password):
            session['logged_in'] = True
            session['username'] = 'admin'
            return index()

        # Student login attempt
        elif Student.login(username, password):
            session['logged_in'] = True
            session['username'] = username
            session['password'] = password
            session['user_type'] = 'student'
            return redirect(url_for('student_dashboard'))

        elif Teacher.login(username, password):
            session['logged_in'] = True
            session['username'] = username
            session['password'] = password
            session['user_type'] = 'teacher'
            return redirect(url_for('teacher_dashboard'))

        else:
            error = "Invalid username or password. Please try again."
            return render_template('login.html', error=error)

    return render_template('login.html')


@app.route('/student_dashboard')
def student_dashboard():
    courses = read_courses_from_csv()
    return render_template('student_dashboard.html', courses=courses)

@app.route('/feedback')
def feedback():
    return render_template('feedback.html', course_data=course_data)

if not os.path.isfile(csv_filename_feedback_records):
    with open(csv_filename_feedback_records, 'w', newline='') as csvfile:
        fieldnames = ['Timestamp', 'Course ID', 'Learning Outcomes', 'Workload Manageable', 'Well Organized',
                      'Student Participation', 'Student Progress', 'Student Attendance',
                      'Well Structured', 'Encouraged Participation', 'Conducive to Learning',
                      'Conducive to Learning','Learning Materials', 'Recommended Reading', 'Library Resources', 'Online Resources',
                      'Interest Stimulated', 'Course Pace', 'Clear Presentation',
                      'Assessment Reasonable', 'Feedback Timely', 'Feedback Helpful','BestFeatures','ImprovementAreas']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


@app.route('/TEACHER_EVALUATION_FORM/<course_id>', methods=['GET', 'POST'])
def TEACHER_EVALUATION_FORM(course_id):
    teacher_details = get_teacher_details(course_id)
    return render_template('TEACHER_EVALUATION_FORM.html', course_id=course_id, teacher_details=teacher_details)

def calculate_sum(ratings):
    total = 0
    for rating in ratings:
        if rating:
            total += int(rating)
    return total


@app.route('/submit_teacher_evaluation/<course_id>', methods=['POST'])
def submit_teacher_evaluation(course_id):

    data = {
        'Course ID': course_id,
        'punctuality': request.form.get('punctuality'),
        'regularity': request.form.get('regularity'),
        'attendance': request.form.get('attendance'),
        'syllabus': request.form.get('syllabus'),
        'organization': request.form.get('organization'),
        'arrangement': request.form.get('arrangement'),
        'focusSyllabi': request.form.get('focusSyllabi'),
        'selfConfidence': request.form.get('selfConfidence'),
        'communicationSkills': request.form.get('communicationSkills'),
        'classroomDiscussions': request.form.get('classroomDiscussions'),
        'teachingSubjectMatter': request.form.get('teachingSubjectMatter'),
        'structuredLecture': request.form.get('structuredLecture'),
        'linkingSubject': request.form.get('linkingSubject'),
        'latestDevelopments': request.form.get('latestDevelopments'),
        'teachingAids': request.form.get('teachingAids'),
        'whiteboardWork': request.form.get('whiteboardWork'),
        'innovativeTeaching': request.form.get('innovativeTeaching'),
        'sharesAnswers': request.form.get('sharesAnswers'),
        'evaluatedAnswerBooks': request.form.get('evaluatedAnswerBooks'),
        'beingUnderstood': request.form.get('beingUnderstood'),
        'academicInterests': request.form.get('academicInterests'),
        'studyMaterial': request.form.get('studyMaterial'),
        'ethnicity': request.form.get('ethnicity'),
        'gender': request.form.get('gender'),
        'challenges': request.form.get('challenges'),
        'professionalSkills': request.form.get('professionalSkills'),
        'careerGoals': request.form.get('careerGoals'),
        'strengths': request.form.get('strengths'),
        'controlMechanism': request.form.get('controlMechanism'),
        'studentParticipation': request.form.get('studentParticipation'),
        'behaviorSkills': request.form.get('behaviorSkills'),
        'opinionTendency': request.form.get('opinionTendency'),
        'reinforcementMechanism': request.form.get('reinforcementMechanism'),
        'ethicalConduct': request.form.get('ethicalConduct'),
        'roleModel': request.form.get('roleModel'),
        'additionalRemarks': request.form.get('additionalRemarks'),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    write_to_csv(data)
    return render_template('thankyoupage.html')


@app.route('/available_courses')
def available_courses():
    df_users_courses = pd.read_csv('assignment_records.csv')

    courses_info = df_users_courses.groupby('Course ID').agg({'username': lambda x: list(x), 'course_name': 'first'}).reset_index()

    return render_template('available_courses.html', courses_info=courses_info)

@app.route('/view_data2/<int:course_id>')
def view_data2(course_id):
    df_feedback = pd.read_csv('teacher_evaluations.csv')
    df_users_courses = pd.read_csv('assignment_records.csv')

    df_merged = pd.merge(df_feedback, df_users_courses, on='Course ID')
    df_merged = df_merged.drop_duplicates(subset='Course ID')

    df_selected_course = df_merged[df_merged['Course ID'] == course_id]

    tables = {
        'Table A': ['punctuality', 'regularity', 'attendance', 'syllabus', 'organization', 'arrangement'],
        'Table B': ['focusSyllabi', 'selfConfidence', 'communicationSkills', 'classroomDiscussions',
                    'teachingSubjectMatter', 'structuredLecture', 'linkingSubject', 'latestDevelopments'],
        'Table C': ['teachingAids', 'whiteboardWork', 'innovativeTeaching', 'sharesAnswers',
                    'evaluatedAnswerBooks', 'beingUnderstood'],
        'Table D': ['academicInterests', 'studyMaterial', 'ethnicity', 'gender', 'challenges',
                    'professionalSkills', 'careerGoals', 'strengths'],
        'Table E': ['controlMechanism', 'studentParticipation', 'behaviorSkills', 'opinionTendency',
                    'reinforcementMechanism', 'ethicalConduct', 'roleModel']
    }

    df_selected_course = df_selected_course.apply(pd.to_numeric, errors='ignore')

    plots = []

    for table_name, columns_of_interest in tables.items():
        mean_scores = df_selected_course[columns_of_interest].mean()

        fig = px.bar(mean_scores, x=mean_scores.index, y=mean_scores.values,
                     title=f'Mean Feedback Scores for {table_name} (Course ID: {course_id})',
                     labels={'x': 'Feedback Aspect', 'y': 'Mean Score'})

        plots.append(fig.to_html(full_html=False))

    return render_template('view_data2.html', plots=plots)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if token.strip()]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    try:

        course_id = request.form['course_id']
        learning_outcomes = request.form['learningOutcomes']
        workload_manageable = request.form['workloadManageable']
        well_organized = request.form['wellOrganized']
        student_participation = request.form['studentParticipation']
        student_progress = request.form['studentProgress']
        student_attendance = request.form['studentAttendance']

        well_structured = request.form['wellStructured']
        encouraged_participation = request.form['encouragedParticipation']
        conducive_to_learning = request.form['conduciveToLearning']

        learning_materials = request.form['learningMaterials']
        recommended_reading = request.form['recommendedReading']
        library_resources = request.form['libraryResources']
        online_resources = request.form['onlineResources']

        interest_stimulated = request.form['interestStimulated']
        course_pace = request.form['coursePace']
        clear_presentation = request.form['clearPresentation']

        assessment_reasonable = request.form['assessmentReasonable']
        feedback_timely = request.form['feedbackTimely']
        feedback_helpful = request.form['feedbackHelpful']

        best_features = request.form.get('bestFeatures', '')
        improvement_areas = request.form.get('improvementAreas', '')

        best_features_cleaned = preprocess_text(best_features)
        improvement_areas_cleaned = preprocess_text(improvement_areas)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')




        with open(csv_filename_feedback_records, 'a', newline='') as csvfile:
            fieldnames = ['Timestamp', 'Course ID', 'Learning Outcomes', 'Workload Manageable', 'Well Organized',
                          'Student Participation', 'Student Progress', 'Student Attendance',
                          'Well Structured', 'Encouraged Participation',
                          'Conducive to Learning','Learning Materials', 'Recommended Reading', 'Library Resources', 'Online Resources',
                          'Interest Stimulated', 'Course Pace', 'Clear Presentation',
                          'Assessment Reasonable', 'Feedback Timely', 'Feedback Helpful','BestFeatures','ImprovementAreas']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if csvfile.tell() == 0:
                writer.writeheader()

            writer.writerow({
                'Timestamp': timestamp,
                'Course ID': course_id,
                'Learning Outcomes': learning_outcomes,
                'Workload Manageable': workload_manageable,
                'Well Organized': well_organized,

                'Student Participation': student_participation,
                'Student Progress': student_progress,
                'Student Attendance': student_attendance,
                'Well Structured': well_structured,
                'Encouraged Participation': encouraged_participation,

                'Conducive to Learning': conducive_to_learning,
                'Learning Materials': learning_materials,
                'Recommended Reading': recommended_reading,
                'Library Resources': library_resources,
                'Online Resources': online_resources,

                'Interest Stimulated': interest_stimulated,
                'Course Pace': course_pace,
                'Clear Presentation': clear_presentation,

                'Assessment Reasonable': assessment_reasonable,
                'Feedback Timely': feedback_timely,
                'Feedback Helpful': feedback_helpful,
                'BestFeatures': best_features_cleaned,
                'ImprovementAreas': improvement_areas_cleaned
            })
        if check_attendance(student_attendance):
            return redirect(url_for('TEACHER_EVALUATION_FORM', course_id=request.form['course_id']))
        else:
            return render_template('/Thankyoufeed_back.html')




    except Exception as e:
        error_message = f'Error occurred: {str(e)}'
        print(error_message)
        flash(error_message, 'error')
        return render_template('loginerror.html')

@app.route('/submitted_feedback')
def submitted_feedback():
    return render_template('submitted_feedback.html')



@app.route('/thank_you')
def thank_you():
    return render_template('Thankyoufeed_back.html')



@app.route('/get_course_details/<course_id>')
def get_course_details(course_id):
    if course_id in course_data:
        return jsonify(course_data[course_id])
    else:
        return jsonify({'error': 'Course ID not found'})


@app.route('/enroll/<course_id>')
def enroll(course_id):

    student_username = session.get('username')
    if not student_username:
        return "Error: No username provided!"

    result = update_enrollment_status(course_id, student_username)
    return result



@app.route('/teacher_dashboard')
def teacher_dashboard():
    username = session.get('username')

    if not username:
        return redirect(url_for('login'))

    df_assignment_records = pd.read_csv('assignment_records.csv')
    courses_taught = df_assignment_records[df_assignment_records['username'] == username]
    courses_info = courses_taught.to_dict(orient='records')

    return render_template('teacher_dashboard.html', courses_info=courses_info)

def calculate_sentiment_score(feedback):
    feedback_dict = {}
    for pair in feedback.split(', '):
        key, value = pair.split(': ')
        feedback_dict[key.strip()] = int(value)

    score_sum = sum(feedback_dict.values())

    normalized_score = max(-2, min(2, score_sum / len(feedback_dict)))

    return normalized_score

@app.route('/teacher_view_data')
def teacher_view_data():
    df_feedback = pd.read_csv('feedback_data.csv')
    df_users_courses = pd.read_csv('assignment_records.csv')

    df_merged = pd.merge(df_feedback, df_users_courses, on='Course ID')

    username = session.get('username')
    teacher_courses = df_merged[df_merged['username'] == username]

    teacher_courses['Course ID'] = teacher_courses['Course ID'].astype(str)

    columns_of_interest = ['Learning Outcomes', 'Workload Manageable', 'Well Organized',
                           'Student Participation', 'Student Progress',
                           'Well Structured', 'Encouraged Participation', 'Conducive to Learning',
                           'Learning Materials', 'Recommended Reading', 'Library Resources', 'Online Resources',
                           'Interest Stimulated', 'Course Pace', 'Clear Presentation',
                           'Assessment Reasonable', 'Feedback Timely', 'Feedback Helpful']

    teacher_courses[columns_of_interest] = teacher_courses[columns_of_interest].apply(pd.to_numeric, errors='coerce')

    plots = []
    course_feedbacks = []

    for _, course_data in teacher_courses.groupby('Course ID'):
        course_mean = course_data[columns_of_interest].mean()
        course_mean_dict = course_mean.to_dict()

        feedback_list = []
        for col, mean_score in course_mean_dict.items():
            feedback_list.append({'category': col, 'mean_score': mean_score})

        course_feedbacks.append({'course_name': course_data['course_name'].iloc[0], 'feedback': feedback_list})

    return render_template('teacher_view_data.html', course_feedbacks=course_feedbacks)



@app.route('/add_faculty', methods=['POST'])
def add_faculty_route():
    faculty_name = request.form['faculty_name']
    faculty_password = request.form['faculty_password']
    result = add_faculty_to_csv(faculty_name, faculty_password)
    return result

@app.route('/update_faculty', methods=['POST'])
def update_faculty_route():
    faculty_name = request.form['faculty_name']
    faculty_password = request.form['faculty_password']
    new_faculty_name = request.form['new_faculty_name']
    new_faculty_password = request.form['new_faculty_password']
    result = update_faculty_in_csv(faculty_name, faculty_password, new_faculty_name, new_faculty_password)
    return result

@app.route('/delete_faculty', methods=['POST'])
def delete_faculty_route():
    username = request.form['delete_faculty_name']
    result = delete_faculty_from_csv(username)
    return result
    return render_template('thankyoupage.html')


@app.route('/manage_course', methods=['POST'])
def manage_course():
    if request.method == 'POST':
        action = request.form['action']

        if action == 'add':
            course_name = request.form['course_name']
            add_course_to_csv(course_name)
            return "Course added successfully"

        elif action == 'update':
            course_name = request.form['course_name']
            new_course_name = request.form['new_course_name']
            update_course_in_csv(course_name, new_course_name)
            return "Course updated successfully"

        elif action == 'delete':
            course_name = request.form['course_name']
            delete_course_from_csv(course_name)
            return "Course deleted successfully"

    return "Invalid request"


@app.route('/logout', methods=['POST'])
def logout():
    session['logged_in'] = False
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(port=5978, host='0.0.0.0', debug=True)
