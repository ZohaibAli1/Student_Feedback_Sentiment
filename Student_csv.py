import csv
from flask import Flask, render_template

students_courses_file = 'students_courses.csv'
csv_filename_students = 'students.csv'
csv_filename_assignment_records = 'assignment_records.csv'

def read_courses_from_csv():
    courses = []
    with open(csv_filename_assignment_records, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            courses.append(row)
    return courses

def update_enrollment_status(course_id, student_username):
    enrollments = []
    with open('students_courses.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        enrollments = list(reader)

    is_enrolled = False
    for enrollment in enrollments:
        if enrollment['course_id'] == course_id and enrollment['student_username'] == student_username:
            is_enrolled = True
            break

    if not is_enrolled:
        new_enrollment = {
            'course_id': course_id,
            'student_username': student_username,
        }
        enrollments.append(new_enrollment)

    fieldnames = ['course_id', 'student_username']
    with open('students_courses.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enrollments)

    return render_template('student_backhome.html')


class Student:
    @staticmethod
    def read_csv(filename):
        data = []
        with open(filename, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        return data

    @staticmethod
    def login(username, password):
        students_data = Student.read_csv(csv_filename_students)
        for student in students_data:
            if student['student_username'] == username and student['password'] == password:
                return True
        return False

    @classmethod
    def check_student(cls, username):
        pass


def add_student_to_csv(student_username, student_password, csv_filename_students):
    with open(csv_filename_students, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[0] == student_username:
                return render_template('Already_register.html')


    with open(csv_filename_students, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([student_username, student_password])
    return render_template('success-page.html')


def update_student_in_csv(student_username, student_password,new_student_name,new_student_password):
    student_records = []

    with open(csv_filename_students, mode='r') as file:
        reader = csv.reader(file)
        student_records = list(reader)

    updated = False
    for record in student_records:
        if record[0] == student_username and record[1] == student_password:
            record[0] = new_student_name
            record[1] = new_student_password
            updated = True

    if updated:
        with open(csv_filename_students, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(student_records)
        return render_template('update.html')
    else:
        return "Student not found"


def delete_student_from_csv(student_username):
    student_records = []

    with open(csv_filename_students, mode='r') as file:
        reader = csv.reader(file)
        student_records = list(reader)

    updated_records = [record for record in student_records if record[0] != student_username]

    with open(csv_filename_students, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(updated_records)

    return render_template('delete_html.html')
