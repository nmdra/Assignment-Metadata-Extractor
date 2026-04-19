import argparse
import json
import random
from pathlib import Path

STUDENT_NUMBER_KEYS = [
    "Student No",
    "Stu. ID",
    "Student ID",
    "ID",
    "Reg No",
    "Registration No",
    "Index No",
    "Reg. Number",
]
STUDENT_NAME_KEYS = ["Name", "Full Name", "Student Name", "Student", "Stu. Name"]
ASSIGNMENT_KEYS = [
    "Assignment #",
    "Assignment No",
    "HW",
    "Task No",
    "Submission No",
    "Assgn #",
    "Worksheet No",
]
SEPARATORS = [": ", " - ", " = ", ": "]
LINE_BREAKS = ["\n", " | ", ", ", "  "]

NAMES = [
    "Amal Perera",
    "Nimal Silva",
    "Kasun Fernando",
    "Dilini Rathnayake",
    "Chamara Bandara",
    "Sithum Jayawardena",
    "Amali Gunasekara",
]


def make_example(student_num: int, name: str, assign_num: int) -> dict:
    sk = random.choice(STUDENT_NUMBER_KEYS)
    nk = random.choice(STUDENT_NAME_KEYS)
    ak = random.choice(ASSIGNMENT_KEYS)
    sep = random.choice(SEPARATORS)
    lb = random.choice(LINE_BREAKS)
    text = f"{sk}{sep}{student_num}{lb}{nk}{sep}{name}{lb}{ak}{sep}{assign_num}"
    return {
        "instruction": "Extract student info as JSON from the following text.",
        "input": text,
        "output": json.dumps(
            {
                "student_number": str(student_num),
                "student_name": name,
                "assignment_number": str(assign_num),
            }
        ),
    }


def generate_dataset(size: int) -> list[dict]:
    dataset = [
        make_example(20210000 + i, random.choice(NAMES), (i % 10) + 1) for i in range(size)
    ]
    random.shuffle(dataset)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate student extraction dataset.")
    parser.add_argument("--size", type=int, default=400, help="Number of examples to generate.")
    parser.add_argument(
        "--output",
        default="data/dataset.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = generate_dataset(args.size)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {len(dataset)} examples at {output_path}")


if __name__ == "__main__":
    main()
