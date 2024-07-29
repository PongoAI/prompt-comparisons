import csv
import os


def calculate_single_true_false_ratio(filename=None):
    true_count = 0
    false_count = 0

    if filename:
        files = [filename]
    else:
        files = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith('.csv')]

    for file in files:
        with open(file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                relevance = row['relevance_assessment']


                if relevance.lower() == 'true':
                    true_count += 1
                elif relevance.lower() == 'false':
                    false_count += 1
                else:
                    print(relevance)
                    print('relevance')

    if false_count == 0:
        return "All assessments are True"
    else:
        return true_count / (false_count + true_count)




print(calculate_single_true_false_ratio('your_file.csv'))
