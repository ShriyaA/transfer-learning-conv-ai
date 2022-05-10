import json
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path of the file containing unfiltered questions")
    args = parser.parse_args()

    questions = []
    filtered_items = []


    with open(args.input_file) as f:
        item = f.readline()
        read_questions = 0
        while(item):
            read_questions += 1
            item = json.loads(item)

            question = item['question'].strip()
            answer = item['orig_answer']['text'].strip()

            if len(question) > 1 and question[-1] == '?' and len(answer) > 15 and question not in questions:
                filtered_items.append(item)
                questions.append(question.strip())

            item = f.readline()

    print(f"{read_questions} questions read from input file")
    print(f"{len(filtered_items)} questions retained after filtering")

    with open('generated_questions_filtered.json', 'w') as f:
        json.dump(filtered_items, f)

if __name__=='__main__':
    main()