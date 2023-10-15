import csv

input_csv_path = 'data2.csv'
output_csv_path = 'data2_with_all_numerical_features.csv'

input_csv_file = open(input_csv_path, newline='')
output_csv_file = open(output_csv_path, 'w', newline='')

input_csv_reader = csv.reader(input_csv_file)
output_csv_writer = csv.writer(output_csv_file)

for row in input_csv_reader:
    if row[4] == 'Present':
        row[4] = '1'
    elif row[4] == 'Absent':
        row[4] = '0'
    else:
        raise ValueError("The value of the 5th element of the row should be either \"Present\" or \"Absent\"")
    output_csv_writer.writerow(row)

