import csv
from flask import render_template
csv_filename_courses = 'courses.csv'

def add_course_to_csv(course_id, course_name):
    with open(csv_filename_courses, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([course_id, course_name])
    return render_template('success-page.html')

def update_course_in_csv(course_id, new_course_name):
    course_records = []

    with open(csv_filename_courses, mode='r') as file:
        reader = csv.reader(file)
        course_records = list(reader)

    updated = False
    for record in course_records:
        if record[0] == course_id:
            record[1] = new_course_name
            updated = True

    if updated:
        with open(csv_filename_courses, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(course_records)
        return render_template('update.html')
    else:
        return "Course not found"

def delete_course_from_csv(course_id):
    course_records = []

    with open(csv_filename_courses, mode='r') as file:
        reader = csv.reader(file)
        course_records = list(reader)

    updated_records = [record for record in course_records if record[0] != course_id]

    with open(csv_filename_courses, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(updated_records)
    return render_template('delete_html.html')