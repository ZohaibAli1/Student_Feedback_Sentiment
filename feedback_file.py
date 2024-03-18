import csv

csv_filename_assignment_records = 'assignment_records.csv'
csv_filename_feedback_records = 'feedback_data.csv'


course_data = {}

with open('assignment_records.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        course_data[row['Course ID']] = {
            'username': row['username'],
            'course_name': row['course_name']
        }

def get_username_from_course_id(course_id):
    assignment_records_file = 'assignment_records.csv'

    course_username_map = {}

    with open(assignment_records_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            course_username_map[row['course_id']] = row['username']

    return course_username_map.get(course_id)