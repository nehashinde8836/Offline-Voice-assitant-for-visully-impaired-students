"""Replays the exact failing session to verify all quiz bugs are fixed."""
import sys; sys.path.insert(0, ".")
import backend.server as srv

def reset():
    srv.state.update({
        "mode": "general", "last_response": "",
        "quiz_session": None, "awaiting_quiz_answer": False,
        "last_a": None, "last_b": None, "last_op": None,
    })

def chat(text, expected_contains=None, expected_not_contains=None):
    r = srv.process_text(text)
    short = r[:80].replace("\n", " | ")
    ok = True
    if expected_contains:
        for s in expected_contains:
            if s not in r:
                print(f"  FAIL: expected '{s}' in response")
                ok = False
    if expected_not_contains:
        for s in expected_not_contains:
            if s in r:
                print(f"  FAIL: did NOT expect '{s}' in response")
                ok = False
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] '{text}' → {short}")
    return r

print("=== Bug 1: 'पाच अधिक तीन' should NOT be treated as quiz answer ===")
reset()
# Simulate leftover quiz from previous session — this is the core bug
# When server restarts, state resets. But within a session:
srv.state["mode"] = "pythagoras"
chat("क्विझ सुरू कर")  # start pyth quiz
assert srv.state["awaiting_quiz_answer"], "Quiz should be active"
r = chat("पाच अधिक तीन",
         expected_not_contains=["चुकीचे उत्तर", "बरोबर उत्तर"],
         expected_contains=["मिळते", "बेरीज", "आहे"])  # should calculate, not quiz answer
assert not srv.state["awaiting_quiz_answer"], "Quiz should have exited"
print("  PASS: arithmetic command exits quiz and calculates")

print()
print("=== Bug 2: 'क्विझ थांब' / 'क्विझ थांबव' should STOP, not restart ===")
reset()
srv.state["mode"] = "pythagoras"
chat("क्विझ सुरू कर")
assert srv.state["awaiting_quiz_answer"]

r = chat("क्विझ थांब", expected_contains=["थांबवली"], expected_not_contains=["सुरू!"])
assert not srv.state["awaiting_quiz_answer"], "Quiz should be stopped"
assert srv.state["quiz_session"] is None, "Quiz session should be None"
print("  PASS: क्विझ थांब stops quiz")

reset()
srv.state["mode"] = "pythagoras"
chat("क्विझ सुरू कर")
r = chat("क्विझ थांबव", expected_contains=["थांबवली"], expected_not_contains=["सुरू!"])
assert not srv.state["awaiting_quiz_answer"]
print("  PASS: क्विझ थांबव stops quiz")

print()
print("=== Bug 3: Mode switch during quiz should exit quiz AND switch mode ===")
reset()
srv.state["mode"] = "pythagoras"
chat("क्विझ सुरू कर")
assert srv.state["awaiting_quiz_answer"]

chat("अंकगणित", expected_contains=["अंकगणित मोड"])
assert not srv.state["awaiting_quiz_answer"], "Quiz should exit on mode switch"
assert srv.state["mode"] == "arithmetic", f"Mode should be arithmetic, got {srv.state['mode']}"
print("  PASS: 'अंकगणित' exits quiz and switches mode")

reset()
srv.state["mode"] = "arithmetic"
chat("क्विझ सुरू कर")
chat("पायथागोरस मोड", expected_contains=["पायथागोरस मोड"])
assert not srv.state["awaiting_quiz_answer"], "Quiz should exit on mode switch"
assert srv.state["mode"] == "pythagoras"
print("  PASS: 'पायथागोरस मोड' exits quiz and switches mode")

print()
print("=== Bug 4: 'परत सांगा' during quiz should repeat question ===")
reset()
srv.state["mode"] = "pythagoras"
chat("क्विझ सुरू कर")
q_text = srv.state["quiz_session"].current_question["question_marathi"]
r = chat("परत सांगा", expected_contains=["प्रश्न पुन्हा"])
assert srv.state["awaiting_quiz_answer"], "Quiz should still be active after repeat"
print("  PASS: 'परत सांगा' repeats question, stays in quiz")

print()
print("=== Bug 5: Arithmetic expressions should NOT be quiz answers ===")
reset()
srv.state["mode"] = "pythagoras"
chat("क्विझ सुरू कर")
assert srv.state["awaiting_quiz_answer"]

# "6 गुणिले 7" has numbers but is an arithmetic command → should exit quiz
r = chat("6 गुणिले 7",
         expected_not_contains=["चुकीचे उत्तर"],
         expected_contains=["गुणाकार", "मिळते"])
assert not srv.state["awaiting_quiz_answer"], "Quiz should exit on arithmetic command"
print("  PASS: '6 गुणिले 7' exits quiz and calculates")

reset()
srv.state["mode"] = "pythagoras"
chat("क्विझ सुरू कर")
r = chat("दहा वजा चार",
         expected_not_contains=["चुकीचे उत्तर"],
         expected_contains=["वजा", "मिळते"])
assert not srv.state["awaiting_quiz_answer"]
print("  PASS: 'दहा वजा चार' exits quiz and calculates")

print()
print("=== Bug 6: 'क्विझ सुरू कर' mid-quiz should restart ===")
reset()
srv.state["mode"] = "pythagoras"
chat("क्विझ सुरू कर")
# Answer one question
q = srv.state["quiz_session"].current_question
chat(str(q["answer"]))
# Now say क्विझ सुरू कर again
r = chat("क्विझ सुरू कर", expected_contains=["सुरू!"])
assert srv.state["awaiting_quiz_answer"], "New quiz should be active"
assert srv.state["quiz_session"].current == 1, "Should be at question 1"
print("  PASS: 'क्विझ सुरू कर' mid-quiz restarts quiz")

print()
print("All quiz bug tests passed!")
