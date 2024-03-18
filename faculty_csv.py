import csv
from flask import Flask, render_template
csv_filename_courses_faculty = 'courses_faculty.csv'
csv_filename_faculty = 'faculty.csv'


class Admin:
    @staticmethod
    def login(username, password):
        if username == 'admin' and password == 'admin':
            return True
        return False
class Teacher:
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
        faculty_data = Teacher.read_csv(csv_filename_faculty)
        for faculty_member in faculty_data:
            if faculty_member['username'] == username and faculty_member['password'] == password:
                return True
        return False

def add_faculty_to_csv(faculty_name, faculty_password):
    with open(csv_filename_faculty, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([faculty_name, faculty_password])
    return render_template('success-page.html')

def update_faculty_in_csv(faculty_name, faculty_password, new_faculty_name, new_faculty_password):
    faculty_records = []

    with open(csv_filename_faculty, mode='r') as file:
        reader = csv.reader(file)
        faculty_records = list(reader)

    updated = False
    for record in faculty_records:
        if record[0] == faculty_name:
            record[0] = new_faculty_name
            record[1] = faculty_password
            record[1] = new_faculty_password
            updated = True

    if updated:
        with open(csv_filename_faculty, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(faculty_records)
        return render_template('update.html')
    else:
        return "Faculty not found"

def delete_faculty_from_csv(faculty_name):
    faculty_records = []

    with open(csv_filename_faculty, mode='r') as file:
        reader = csv.reader(file)
        faculty_records = list(reader)

    updated_records = [record for record in faculty_records if record[0] != faculty_name]

    with open(csv_filename_faculty, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(updated_records)
    return render_template('delete_html.html')



