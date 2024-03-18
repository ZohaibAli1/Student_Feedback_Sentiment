import csv
from flask import render_template


csv_filename_faculty = 'faculty.csv'
csv_filename_assignment_records = 'assignment_records.csv'
csv_filename_courses = 'courses.csv'

def get_available_courses():
    available_courses = []
    with open(csv_filename_courses, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            available_courses.append(row)
    return available_courses

def assign_course_to_teacher(teacher_username, course_id, course_name, csv_filename_faculty, csv_filename_assignment_records):
    global teacher
    faculty_data = []

    with open(csv_filename_faculty, 'r') as file:
        reader = csv.DictReader(file)
        faculty_data = list(reader)

    for teacher in faculty_data:
        if 'course_id' not in teacher:
            teacher['course_id'] = []

        if course_id in teacher['course_id']:
            return "Error: Course already assigned to another teacher."
    teacher_found = False
    for teacher in faculty_data:
        if teacher['username'] == teacher_username:
            teacher['course_id'].append(course_id)
            teacher_found = True
            break

    if teacher_found:
        with open(csv_filename_assignment_records, 'a', newline='') as assignment_file:
            fieldnames = ['username', 'Course ID', 'course_name']
            writer = csv.DictWriter(assignment_file, fieldnames=fieldnames)
            writer.writerow({'username': teacher_username, 'Course ID': course_id, 'course_name': course_name})

        return render_template('teacher_assign.html')

    return "Error: Teacher not found."
