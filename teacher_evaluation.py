import csv
import os
from datetime import datetime

csv_filename_assignment_records = 'assignment_records.csv'
csv_filename_teacher_form ='teacher_evaluation.csv'



def get_teacher_details(course_id):
    # Read teacher details from CSV based on course_id
    with open(csv_filename_assignment_records, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Course ID'] == course_id:
                return row

    return None

def write_to_csv(data):
    fieldnames = ['Course ID', 'punctuality', 'regularity', 'attendance', 'syllabus',
                  'organization', 'arrangement', 'focusSyllabi', 'selfConfidence', 'communicationSkills',
                  'classroomDiscussions', 'teachingSubjectMatter', 'structuredLecture', 'linkingSubject',
                  'latestDevelopments', 'teachingAids', 'whiteboardWork', 'innovativeTeaching', 'sharesAnswers',
                  'evaluatedAnswerBooks', 'beingUnderstood', 'academicInterests', 'studyMaterial', 'ethnicity',
                  'gender', 'challenges', 'professionalSkills', 'careerGoals', 'strengths', 'controlMechanism',
                  'studentParticipation', 'behaviorSkills', 'opinionTendency', 'reinforcementMechanism',
                  'ethicalConduct', 'roleModel', 'additionalRemarks', 'timestamp']

    file_exists = os.path.isfile('teacher_evaluations.csv')

    with open('teacher_evaluations.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)
# def calculate_total(form, category):
#     total = 0
#
#     if category == 'E':
#         for i in range(35, 42):  # Assuming 7 items for 'E' category
#             key = f'{category.lower()}_{i}'
#             if key in form and form[key] != '':
#                 total += int(form[key])
#     else:
#         for i in range(1, 29):  # Assuming 28 items for other categories
#             key = f'{category.lower()}_{i}'
#             if key in form and form[key] != '':
#                 total += int(form[key])
#
#     return total

def check_attendance(attendance_value):
    if attendance_value.startswith('<25'):
        return False
    elif attendance_value.startswith('>75'):
        return True
    elif '26' <= attendance_value.split('%')[0] <= '50':
        return False
    elif '51' <= attendance_value.split('%')[0] <= '75':
        return False
    attendance_percentage = int(attendance_value.strip('%'))
    return attendance_percentage > 75

# def load_data(filename):
#     data = {}
#     with open(filename, 'r', newline='', encoding='utf-8') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             for key, value in row.items():
#                 if key in data:
#                     data[key].append(int(value))
#                 else:
#                     data[key] = [int(value)]
#     return data



# def save_feedback_to_csv(feedback_data, csv_filename_teacher_form):
#     fieldnames = ['Timestamp', 'Course ID', 'Total A', 'Total B', 'Total C', 'Total D', 'Total E','Overall Total']
#
#     is_file_empty = not os.path.isfile(csv_filename_teacher_form) or os.path.getsize(csv_filename_teacher_form) == 0
#
#     with open(csv_filename_teacher_form, 'a', newline='') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # if is_file_empty:
        #     writer.writeheader()
        #
        # writer.writerow(feedback_data)