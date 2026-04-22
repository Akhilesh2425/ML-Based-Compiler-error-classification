import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

from predict import predict_both


class ErrorClassificationAgent:

    def __init__(self):
        pass

    def _read_file(self, file_path: str) -> Optional[str]:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None

        try:
            with open(file_path, "r") as f:
                return f.read()
        except Exception as e:
            print(f"❌ Failed to read file: {e}")
            return None
            
    def _choose_best(self, results: List[Dict]) -> Tuple[Dict, str]:
        """
        Decide which model output to trust (supervised, SVM+AST, BERT when available).
        """

        if not results:
            return {"label": "unknown", "confidence": 0, "model_used": "none"}, "No models available"

        if len(results) == 1:
            return results[0], "Only one model available"

        labels = [r["label"] for r in results]
        unique = set(labels)

        if len(unique) == 1:
            best = max(results, key=lambda x: x["confidence"])
            return best, "All models agree → selecting highest confidence"

        if len(results) >= 3:
            votes = Counter(labels)
            top_label, top_count = votes.most_common(1)[0]
            if top_count >= 2:
                agreeing = [r for r in results if r["label"] == top_label]
                best = max(agreeing, key=lambda x: x["confidence"])
                return best, "Majority agreement → strongest confidence among agreeing models"

        if len(results) == 2:
            r1, r2 = results[0], results[1]
            diff = abs(r1["confidence"] - r2["confidence"])
            best = max(results, key=lambda x: x["confidence"])
            if diff < 5:
                return best, "Models disagree (similar confidence) → selecting slightly better"
            return best, "Models disagree → selecting highest confidence"

        best = max(results, key=lambda x: x["confidence"])
        return best, "Models disagree → selecting highest confidence"

    def classify(
        self,
        text: str = None,
        file_path: str = None,
        file_content: str = None
    ) -> Dict:

        print("\n🤖 Agent started...")


        if file_content:
            print("📂 Mode: FILE CONTENT INPUT")
            results = predict_both(text=text, file_content=file_content)

        elif file_path:
            print("📂 Mode: FILE PATH INPUT")
            content = self._read_file(file_path)

            if not content:
                return self._error_response("File could not be read")

            results = predict_both(file_content=content)

        elif text:
            print("📝 Mode: TEXT INPUT")
            results = predict_both(text=text)

        else:
            return self._error_response("No input provided")


        best, decision = self._choose_best(results)

        return self._process_result(best, decision, results)

    def _process_result(self, result: Dict, decision: str, all_results: List[Dict]) -> Dict:
        label = result["label"]
        confidence = result["confidence"]

        print(f"🎯 Final Decision: {label} ({confidence}%)")

        advice = self._generate_advice(label)

        return {
            "final_label": label,
            "confidence": confidence,
            "model_used": result["model_used"],
            "agent_decision": decision,
            "low_confidence": confidence < 60,
            "advice": advice,
            "all_model_outputs": all_results  
        }

    def _generate_advice(self, label: str) -> str:
        label = label.lower()

        if label == "lexical":
            return (
                "Check for invalid tokens, stray characters, or malformed literals. "
                "Look for typos like missing quotes or illegal symbols."
            )

        elif label in ["syntax", "syntactic"]:
            return (
                "Check code structure: missing semicolons, brackets, or incorrect syntax. "
                "Ensure proper statement formation."
            )

        elif label == "semantic":
            return (
                "Check variable declarations, types, and function usage. "
                "Look for undeclared variables or type mismatches."
            )

        return "No specific advice available."

    def _error_response(self, message: str) -> Dict:
        return {
            "final_label": "unknown",
            "confidence": 0,
            "model_used": "none",
            "agent_decision": "Error",
            "low_confidence": True,
            "advice": message,
            "all_model_outputs": []
        }


if __name__ == "__main__":
    agent = ErrorClassificationAgent()

    print("\n--- TEST TEXT ---")
    print(agent.classify(text="error: expected ';' before return"))

    print("\n--- TEST FILE ---")
    print(agent.classify(file_path="test.c"))