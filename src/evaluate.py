import inspect
import json

from llmtuner import Evaluator, CalibrationEvaluator

def main():
    evaluator = CalibrationEvaluator()
    evaluator.eval()

if __name__ == "__main__":
    main()