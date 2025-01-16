from typing import TypedDict
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai

app =  FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define Input and Output State
class InputState(TypedDict):
    suggested_answer: str
    student_answer: str

class OutputState(TypedDict):
    gemini_score: float
    feedback: str
    sbert_score: float
    minilm_score:float
    labse_score:float


#---------------------------------------------------------------- 
#                       Gemini Model
#---------------------------------------------------------------- 

def gemini_evaluation_node(state: InputState) -> OutputState:
    """Use Gimmia Generative AI to evaluate the student answer and return llm_score with feedback."""

    # Configure Gimmia Generative AI
    genai.configure(api_key="AIzaSyDUiT3yPTTo2nmoPRj-hpo2r2OyrPH5cqs")
    model = genai.GenerativeModel("gemini-2.0-flash-exp")

    # Prepare the evaluation prompt
    prompt = f"""
    Suggested answer: "{state['suggested_answer']}"
    Student’s answer: "{state['student_answer']}"

    Evaluate how well the student’s answer matches the suggested answer on a scale from 0 to 10,
    considering correctness, completeness, and clarity. Provide a numeric score followed by a one-line feedback.

    Examples:
    1. Suggested answer: "The capital of France is Paris."
       Student’s answer: "Paris is the capital of France."
       Evaluation: "10 - Perfectly correct and well-phrased."

    2. Suggested answer: "The mitochondria is the powerhouse of the cell."
       Student’s answer: "Mitochondria generates energy for the cell."
       Evaluation: "9.5 - Correct but slightly less precise phrasing."

    3. Suggested answer: "The water cycle includes evaporation, condensation, and precipitation."
       Student’s answer: "The water cycle has evaporation, condensation, and precipitation."
       Evaluation: "10 - Matches the suggested answer exactly."

    4. Suggested answer: "The process of photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight."
       Student’s answer: "Photosynthesis uses sunlight to make food for plants."
       Evaluation: "7 - Partially correct but lacks completeness and detail."

    5. Suggested answer: "E=mc^2 explains the relationship between energy, mass, and the speed of light."
       Student’s answer: "Energy equals mass times the speed of light squared."
       Evaluation: "9 - Correct but missing the context of what the equation represents."

    6. Suggested answer: "Solve for x: 2x + 3 = 7."
       Student’s answer: "x = 2."
       Evaluation: "10 - Correct solution with proper calculation."

    7. Suggested answer: "The derivative of x^2 is 2x."
       Student’s answer: "Derivative of x squared is 2x."
       Evaluation: "10 - Correct and matches the mathematical explanation."

    8. Suggested answer: "The area of a circle is calculated as A = πr^2."
       Student’s answer: "Area of a circle equals pi times radius squared."
       Evaluation: "10 - Accurate and clear answer."

    9. Suggested answer: "The sum of angles in a triangle is 180 degrees."
       Student’s answer: "All triangle angles add up to 180 degrees."
       Evaluation: "10 - Correct and well-stated."

    10. Suggested answer: "Integrate x^2 with respect to x."
        Student’s answer: "x^3/3 + C."
        Evaluation: "10 - Correct integration with constant of integration."

    11. Suggested answer: "In 1492, Christopher Columbus sailed across the Atlantic Ocean."
        Student’s answer: "Columbus discovered America in 1492."
        Evaluation: "8 - Partially correct but oversimplified and historically incomplete."

    12. Suggested answer: "World War II began in 1939 and ended in 1945."
        Student’s answer: "World War II ended in 1945."
        Evaluation: "5 - Partially correct but lacks context and detail."

    13. Suggested answer: "Water boils at 100°C under standard atmospheric pressure."
        Student’s answer: "Water boils at 100 degrees Celsius."
        Evaluation: "10 - Perfectly correct and concise."

    14. Suggested answer: "Shakespeare wrote the play Hamlet."
        Student’s answer: "Hamlet was written by Shakespeare."
        Evaluation: "10 - Correct and matches the suggested answer."

    15. Suggested answer: "Newton’s second law states that force equals mass times acceleration (F=ma)."
        Student’s answer: "Force equals mass times acceleration."
        Evaluation: "10 - Correct but could include the formula explicitly for completeness."

    Suggested answer: "{state['suggested_answer']}"
    Student’s answer: "{state['student_answer']}"
    """

    try:
        # Generate content using the model
        response = model.generate_content(prompt)
        output = response.text.strip()

        if " - " in output:
            score, feedback = output.split(" - ", 1)
            score = float(score)
        else:
            score = 0.0
            feedback = "No valid feedback provided."

        print(f"LLM Evaluation Result: score={score}, feedback={feedback}")
    except Exception as e:
        print(f"Exception in LLM evaluation: {e}")
        score = 0.0
        feedback = "Error in LLM evaluation."

    # Update and return the state
    state["gemini_score"] = score
    state["feedback"] = feedback
    return state


#---------------------------------------------------------------- 
#                   FastAPI BaseModel
#----------------------------------------------------------------
class Item(BaseModel):
    suggested_answer: str
    student_answer: str




#---------------------------------------------------------------------- 
#                     FastAPI endpoint for Hello 
#----------------------------------------------------------------------

@app.post("/hello")
async def Hello():
    return ("hello")

#---------------------------------------------------------------------- 
#                     FastAPI endpoint for Evaluation
#----------------------------------------------------------------------
@app.post("/evaluate")
async def evaluate_items(items: List[Item]):
    states = []  # Initialize a list to store results

    for item in items:
        state = {
            "suggested_answer": item.suggested_answer,
            "student_answer": item.student_answer
        }

        try:
            state =gemini_evaluation_node(state)
            # Append the fully processed state to the results list
            states.append(state)
        except Exception as e:
            print(f"Exception during evaluation for item: {e}")
            # Append a default error result for the failed evaluation
            states.append({
                "suggested_answer": item.suggested_answer,
                "student_answer": item.student_answer,
                "error": "Evaluation failed.",
                "scores": {
                    "gemini_score": 0.0,

                }
            })

    # Return the results list as the API response
    return {"results": states}

#---------------------------------------------------------------- 
#                       Run FastAPI
#----------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7100)
    